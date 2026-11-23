// Map an x64 PE image without LoadLibrary and bind its imports through a host resolver.
// Scope: DIR64/HIGHLOW relocations, ordinary and delay imports by name or ordinal, section protections,
// .pdata registration, TLS callbacks and the entry point. Delay imports are bound eagerly: the
// image's delay-load helper would call LoadLibrary and bypass the host resolver. An implicit TLS
// data template is served through a generated carrier DLL that Windows loads (see TlsCarrier).
// Header fields are checked against the file or SizeOfImage before use.
#include <windows.h>

#include <bcrypt.h>
#include <psapi.h>
#include <sddl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <new>
#include <string>
#include <vector>

// Keep in step with STATUS in tests/cpu/native_map_check.py.
enum MapStatus : std::uint32_t {
    MAP_OK = 0,
    MAP_IO = 1,
    MAP_FORMAT = 2,
    MAP_MEMORY = 3,
    MAP_RELOCATION = 4,
    MAP_UNRESOLVED_IMPORT = 5,
    MAP_UNSUPPORTED = 6,
    MAP_PROTECT = 7,
    MAP_FUNCTION_TABLE = 8,
    MAP_INIT_FAILED = 9,
};

// Returns the address bound to one import, or null to reject it. name is null for an ordinal.
using Resolver = void*(__cdecl*)(const char* dll, const char* name, std::uint32_t ordinal);

namespace {

constexpr std::uint64_t MAX_FILE = 1ull << 30;

struct Failure {
    std::uint32_t* status;
    char* detail;
    std::size_t capacity;

    template <typename... Args>
    void* operator()(MapStatus code, const char* format, Args... args) const {
        if (status) *status = code;
        if (detail && capacity) std::snprintf(detail, capacity, format, args...);
        return nullptr;
    }
};

bool inside(std::uint64_t offset, std::uint64_t size, std::uint64_t limit) {
    return offset <= limit && size <= limit - offset;
}

const IMAGE_DATA_DIRECTORY& directory(const IMAGE_NT_HEADERS64* nt, int index) {
    static const IMAGE_DATA_DIRECTORY empty{};
    return nt->OptionalHeader.NumberOfRvaAndSizes > static_cast<DWORD>(index)
        ? nt->OptionalHeader.DataDirectory[index] : empty;
}

// A view of the mapped image that rejects any range outside SizeOfImage.
struct Image {
    std::uint8_t* base;
    std::uint64_t size;

    template <typename T>
    T* at(std::uint64_t rva, std::uint64_t count = 1) const {
        return inside(rva, sizeof(T) * count, size) ? reinterpret_cast<T*>(base + rva) : nullptr;
    }

    const char* string(std::uint64_t rva) const {
        if (rva >= size) return nullptr;
        const auto* start = base + rva;
        const bool terminated = std::memchr(start, 0, static_cast<std::size_t>(size - rva)) != nullptr;
        return terminated ? reinterpret_cast<const char*>(start) : nullptr;
    }

    const IMAGE_NT_HEADERS64* headers() const {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
        return reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    }
};

DWORD protection(DWORD flags) {
    const bool exec = flags & IMAGE_SCN_MEM_EXECUTE, write = flags & IMAGE_SCN_MEM_WRITE;
    const bool read = flags & IMAGE_SCN_MEM_READ;
    if (exec) return write ? PAGE_EXECUTE_READWRITE : read ? PAGE_EXECUTE_READ : PAGE_EXECUTE;
    return write ? PAGE_READWRITE : read ? PAGE_READONLY : PAGE_NOACCESS;
}

// True if rva lies inside a section mapped as executable.
bool executable(const IMAGE_NT_HEADERS64* nt, std::uint64_t rva) {
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if ((s.Characteristics & IMAGE_SCN_MEM_EXECUTE) && rva >= s.VirtualAddress && rva - s.VirtualAddress < span)
            return true;
    }
    return false;
}

// True if [rva, rva + size) lies inside a section mapped as writable. The Windows loader
// writes the TLS index to AddressOfIndex, so that slot must be writable: measured on
// Windows 11, an index in the headers, .text, .rdata or .tlscb fails with ERROR_NOACCESS
// while another DWORD in writable .data loads. The check is by section characteristics,
// like executable() above, so it holds both after mapping and on release.
bool writable_section(const IMAGE_NT_HEADERS64* nt, std::uint64_t rva, std::uint64_t size) {
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if ((s.Characteristics & IMAGE_SCN_MEM_WRITE) && rva >= s.VirtualAddress
            && size <= span && rva - s.VirtualAddress <= span - size)
            return true;
    }
    return false;
}

// True if every page of [address, address + size) is committed and currently allows reads.
// Image bounds alone do not make bytes readable: section protections apply, and the image may
// change them at run time, so the live protection is queried at each read, including on release.
// The query is not atomic with the read that follows: another thread reprotecting the same pages
// in between is not handled.
bool readable(const void* address, std::size_t size) {
    auto* cursor = static_cast<const std::uint8_t*>(address);
    const auto* end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cursor, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
        const DWORD access = info.Protect & 0xff;
        if ((info.Protect & PAGE_GUARD) || access == PAGE_NOACCESS || access == PAGE_EXECUTE) return false;
        cursor = static_cast<const std::uint8_t*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

// True if the page at address is committed and neither guarded nor PAGE_NOACCESS. Unlike
// readable(), this accepts execute-only pages, which can hold exported code GetProcAddress returns.
bool accessible(const void* address) {
    MEMORY_BASIC_INFORMATION info{};
    if (!VirtualQuery(address, &info, sizeof(info)) || info.State != MEM_COMMIT) return false;
    const DWORD access = info.Protect & 0xff;
    return !(info.Protect & PAGE_GUARD) && access != PAGE_NOACCESS;
}

// A NUL-terminated string inside the image, scanned only across readable committed pages.
// Returns null if the string would require reading past the image or an unreadable page.
const char* readable_string(const Image& image, std::uint64_t rva) {
    if (rva >= image.size) return nullptr;
    const auto* start = image.base + rva;
    for (std::uint64_t scanned = 0; scanned < image.size - rva;) {
        const auto* cursor = start + scanned;
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(cursor, &info, sizeof(info)) || info.State != MEM_COMMIT) return nullptr;
        const DWORD access = info.Protect & 0xff;
        if ((info.Protect & PAGE_GUARD) || access == PAGE_NOACCESS || access == PAGE_EXECUTE) return nullptr;
        const auto offset = static_cast<std::uint64_t>(cursor - static_cast<const std::uint8_t*>(info.BaseAddress));
        const std::uint64_t run = info.RegionSize - offset;
        const std::uint64_t left = image.size - rva - scanned;
        const std::uint64_t span = run < left ? run : left;
        if (std::memchr(cursor, 0, static_cast<std::size_t>(span))) return reinterpret_cast<const char*>(start);
        scanned += span;
    }
    return nullptr;
}

constexpr std::uint64_t MAX_TLS_CALLBACKS = 256;
constexpr const char* TOO_MANY_TLS_CALLBACKS = "more than 256 TLS callbacks";
// dlattrRva: descriptor fields hold RVAs instead of virtual addresses. Every x64 linker sets it.
constexpr DWORD DELAY_IMPORT_RVA_BASED = 1;

// TLS callbacks of a mapped, relocated image in table order. Returns null or the reason the list is rejected.
const char* tls_callbacks(const Image& image, const IMAGE_NT_HEADERS64* nt, std::vector<PIMAGE_TLS_CALLBACK>& out) {
    const auto& dir = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (!dir.VirtualAddress) return nullptr;
    const auto* tls = image.at<IMAGE_TLS_DIRECTORY64>(dir.VirtualAddress);
    if (!tls) return "TLS directory outside image";
    // Sections are already protected here, both after mapping and on release.
    if (!readable(tls, sizeof(IMAGE_TLS_DIRECTORY64))) return "TLS directory not readable";
    // Each value is read once after its check, so the validated value is the one used.
    // The loader assigns the TLS index even for an empty template, so the slot must be
    // validated whenever the directory is present, independent of the callback list.
    const auto base = reinterpret_cast<std::uint64_t>(image.base);
    const std::uint64_t index = tls->AddressOfIndex;
    if (index < base || index - base >= image.size || !image.at<DWORD>(index - base))
        return "TLS index outside image";
    if (!writable_section(nt, index - base, sizeof(DWORD))) return "TLS index outside writable section";
    const std::uint64_t list = tls->AddressOfCallBacks;
    if (!list) return nullptr;
    for (std::uint64_t i = 0; i <= MAX_TLS_CALLBACKS; ++i) {
        const std::uint64_t va = list + 8 * i;
        const auto* slot = va >= base ? image.at<std::uint64_t>(va - base) : nullptr;
        if (!slot) return "TLS callback list outside image";
        if (!readable(slot, sizeof(std::uint64_t))) return "TLS callback list not readable";
        const std::uint64_t callback = *slot;
        if (!callback) return nullptr;
        if (i == MAX_TLS_CALLBACKS) return TOO_MANY_TLS_CALLBACKS;
        if (callback < base || callback - base >= image.size || !executable(nt, callback - base))
            return "TLS callback outside executable section";
        out.push_back(reinterpret_cast<PIMAGE_TLS_CALLBACK>(callback));
    }
    return nullptr;
}

// Notify the image as the Windows loader does for LoadLibrary/FreeLibrary: TLS callbacks in table
// order, then the entry point, with a null reserved argument. Returns the entry point's result.
// Measured on Windows 11 build 26300: LoadLibrary sends nothing to a non-DLL image, neither TLS
// callbacks nor the entry point (for an EXE the entry point is the program itself).
BOOL notify(const Image& image, const IMAGE_NT_HEADERS64* nt, const std::vector<PIMAGE_TLS_CALLBACK>& callbacks,
            DWORD reason) {
    if (!(nt->FileHeader.Characteristics & IMAGE_FILE_DLL)) return TRUE;
    for (auto callback : callbacks) callback(image.base, reason, nullptr);
    if (!nt->OptionalHeader.AddressOfEntryPoint) return TRUE;
    using Entry = BOOL(WINAPI*)(HINSTANCE, DWORD, LPVOID);
    const auto entry = reinterpret_cast<Entry>(image.base + nt->OptionalHeader.AddressOfEntryPoint);
    return entry(reinterpret_cast<HINSTANCE>(image.base), reason, nullptr);
}

// Implicit TLS data. Compiled code reaches a thread-local through the thread's
// ThreadLocalStoragePointer array (gs:[0x58]), indexed by the DWORD at AddressOfIndex. ntdll owns
// that array: it sizes it, assigns the indexes, gives every thread a block per index and frees the
// blocks. pe_map does not touch it. For an image with a non-empty template it writes a small
// carrier DLL to a private directory under the temporary directory (see load_carrier) and loads
// it with LoadLibraryExW. The carrier has no code
// and imports nothing; it holds a TLS directory whose template is a copy of the guest's relocated template
// and whose SizeOfZeroFill and Characteristics (alignment) are the guest's. Windows then assigns
// the carrier a real index and allocates, aligns, initializes and frees one block per thread,
// for threads that already exist and for threads created later, as it would for the guest.
// pe_map stores the carrier's index at the guest's AddressOfIndex, so the guest's accesses land
// in the carrier's blocks. FreeLibrary on the carrier returns the index and the blocks; the
// carrier's file and directory are then deleted through the handles the map call holds.
//
// A process that ends with an image still mapped leaves its carrier directory behind, since
// deletion is only requested on release. pe_map does not remove such directories: nothing but a
// handle establishes that a carrier belongs to the caller.
//
// Differences from a Windows load of the guest: the template is copied once, when the image is
// mapped, so a later write to the template in the image does not reach threads created afterwards;
// and code that assumes index 0 without reading AddressOfIndex cannot be served.
constexpr std::uint64_t MAX_TLS_TEMPLATE = 16ull << 20;

// One map call's carrier and everything it owns. Nothing else is ever deleted: cleanup goes through
// these handles, never through a name or a pattern.
struct TlsCarrier {
    HMODULE module = nullptr;
    HANDLE file = INVALID_HANDLE_VALUE;       // read and delete access, shared for reading only
    HANDLE directory = INVALID_HANDLE_VALUE;  // the private directory that holds the file
};

// Deletes the object behind handle. POSIX semantics remove the name when the handle closes, which
// the directory needs: measured on Windows 10 build 19045, a carrier deleted with the classic
// disposition after FreeLibrary sometimes stays listed after its last handle closes (1 of 10 runs),
// and its directory then cannot be removed (ERROR_DIR_NOT_EMPTY). With POSIX semantics the name was
// gone and the directory removed in 10 of 10 runs. File systems without them get the classic one.
bool dispose(HANDLE handle) {
    FILE_DISPOSITION_INFO_EX posix{FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS};
    if (SetFileInformationByHandle(handle, FileDispositionInfoEx, &posix, sizeof(posix))) return true;
    const DWORD error = GetLastError();
    if (error != ERROR_INVALID_PARAMETER && error != ERROR_NOT_SUPPORTED && error != ERROR_INVALID_FUNCTION)
        return false;
    FILE_DISPOSITION_INFO classic{TRUE};
    return SetFileInformationByHandle(handle, FileDispositionInfo, &classic, sizeof(classic)) != FALSE;
}

// Unloads the carrier, then deletes its file and its directory through the handles it holds.
// Returns null, or the first cleanup step that failed with error holding GetLastError; whatever
// could not be deleted stays where it is.
const char* unload(TlsCarrier& carrier, DWORD& error) {
    const char* failed = nullptr;
    auto note = [&](const char* step) {
        if (!failed) {
            failed = step;
            error = GetLastError();
        }
    };
    if (carrier.module && !FreeLibrary(carrier.module)) note("cannot unload the TLS carrier");
    if (carrier.file != INVALID_HANDLE_VALUE) {
        if (!dispose(carrier.file)) note("cannot delete the TLS carrier file");
        CloseHandle(carrier.file);
    }
    if (carrier.directory != INVALID_HANDLE_VALUE) {
        if (!dispose(carrier.directory)) note("cannot remove the TLS carrier directory");
        CloseHandle(carrier.directory);
    }
    carrier = TlsCarrier{};
    return failed;
}

void unload(TlsCarrier& carrier) {
    DWORD ignored = 0;
    unload(carrier, ignored);
}

constexpr DWORD CARRIER_DATA_RVA = 0x1000, CARRIER_FILE_ALIGN = 0x200, CARRIER_PAGE = 0x1000;

constexpr DWORD align_up(DWORD value, DWORD alignment) { return (value + alignment - 1) & ~(alignment - 1); }

// The carrier's index slot holds this value until Windows assigns an index.
constexpr DWORD CARRIER_NO_INDEX = 0xffffffff;

// A relocatable PE32+ DLL with one data section (template, TLS directory, index, empty import
// table) and a .reloc section for the directory's three addresses. It has no code and no entry
// point. Measured on Windows 10 build 19045: a DLL without an import directory loads but is given
// no TLS index, with or without an entry point; an import directory that holds only the null
// descriptor is enough. index_rva receives the RVA of the carrier's TLS index.
std::vector<std::uint8_t> carrier_image(const std::uint8_t* tls_template, std::size_t length, DWORD zero_fill,
                                        DWORD characteristics, DWORD& index_rva) {
    constexpr std::uint64_t preferred = 0x180000000ull;
    // The directory is 40 bytes at a multiple of 64, so a page boundary never splits it.
    const DWORD directory_offset = align_up(static_cast<DWORD>(length), 64);
    const DWORD index_offset = directory_offset + sizeof(IMAGE_TLS_DIRECTORY64);
    const DWORD import_offset = index_offset + 8;
    const DWORD data_size = import_offset + align_up(static_cast<DWORD>(sizeof(IMAGE_IMPORT_DESCRIPTOR)), 8);
    const DWORD data_raw = align_up(data_size, CARRIER_FILE_ALIGN);
    const DWORD reloc_rva = align_up(CARRIER_DATA_RVA + data_size, CARRIER_PAGE);
    const DWORD reloc_size = sizeof(IMAGE_BASE_RELOCATION) + 4 * sizeof(WORD);
    const DWORD headers = CARRIER_FILE_ALIGN;
    std::vector<std::uint8_t> file(std::size_t{headers} + data_raw + CARRIER_FILE_ALIGN);

    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(file.data());
    dos->e_magic = IMAGE_DOS_SIGNATURE;
    dos->e_lfanew = sizeof(IMAGE_DOS_HEADER);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(file.data() + dos->e_lfanew);
    nt->Signature = IMAGE_NT_SIGNATURE;
    nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
    nt->FileHeader.NumberOfSections = 2;
    nt->FileHeader.SizeOfOptionalHeader = sizeof(IMAGE_OPTIONAL_HEADER64);
    nt->FileHeader.Characteristics = IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_LARGE_ADDRESS_AWARE | IMAGE_FILE_DLL;
    auto& opt = nt->OptionalHeader;
    opt.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
    opt.ImageBase = preferred;
    opt.SectionAlignment = CARRIER_PAGE;
    opt.FileAlignment = CARRIER_FILE_ALIGN;
    opt.MajorOperatingSystemVersion = 6;
    opt.MajorSubsystemVersion = 6;
    opt.SizeOfInitializedData = data_raw + CARRIER_FILE_ALIGN;
    opt.SizeOfImage = reloc_rva + CARRIER_PAGE;
    opt.SizeOfHeaders = headers;
    opt.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_GUI;
    opt.DllCharacteristics = IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE | IMAGE_DLLCHARACTERISTICS_NX_COMPAT
        | IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA;
    opt.SizeOfStackReserve = 0x100000;
    opt.SizeOfStackCommit = 0x1000;
    opt.SizeOfHeapReserve = 0x100000;
    opt.SizeOfHeapCommit = 0x1000;
    opt.NumberOfRvaAndSizes = IMAGE_NUMBEROF_DIRECTORY_ENTRIES;
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS] = {CARRIER_DATA_RVA + directory_offset, sizeof(IMAGE_TLS_DIRECTORY64)};
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_BASERELOC] = {reloc_rva, reloc_size};
    // The import table is only its null descriptor, which the zero-filled buffer already holds.
    opt.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {CARRIER_DATA_RVA + import_offset, sizeof(IMAGE_IMPORT_DESCRIPTOR)};

    auto* sections = IMAGE_FIRST_SECTION(nt);
    std::memcpy(sections[0].Name, ".tls", 4);
    sections[0].Misc.VirtualSize = data_size;
    sections[0].VirtualAddress = CARRIER_DATA_RVA;
    sections[0].SizeOfRawData = data_raw;
    sections[0].PointerToRawData = headers;
    sections[0].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE;
    std::memcpy(sections[1].Name, ".reloc", 6);
    sections[1].Misc.VirtualSize = reloc_size;
    sections[1].VirtualAddress = reloc_rva;
    sections[1].SizeOfRawData = CARRIER_FILE_ALIGN;
    sections[1].PointerToRawData = headers + data_raw;
    sections[1].Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE;

    std::uint8_t* data = file.data() + headers;
    if (length) std::memcpy(data, tls_template, length);
    IMAGE_TLS_DIRECTORY64 tls{};
    tls.StartAddressOfRawData = preferred + CARRIER_DATA_RVA;
    tls.EndAddressOfRawData = tls.StartAddressOfRawData + length;
    tls.AddressOfIndex = preferred + CARRIER_DATA_RVA + index_offset;
    tls.SizeOfZeroFill = zero_fill;
    tls.Characteristics = characteristics;
    std::memcpy(data + directory_offset, &tls, sizeof(tls));
    std::memcpy(data + index_offset, &CARRIER_NO_INDEX, sizeof(CARRIER_NO_INDEX));

    // Base relocations for StartAddressOfRawData, EndAddressOfRawData and AddressOfIndex.
    const DWORD directory_rva = CARRIER_DATA_RVA + directory_offset;
    const IMAGE_BASE_RELOCATION block{directory_rva & ~(CARRIER_PAGE - 1), reloc_size};
    WORD entries[4] = {};  // the fourth entry stays IMAGE_REL_BASED_ABSOLUTE padding
    for (DWORD i = 0; i < 3; ++i)
        entries[i] = static_cast<WORD>((IMAGE_REL_BASED_DIR64 << 12) | ((directory_rva + 8 * i) & (CARRIER_PAGE - 1)));
    std::uint8_t* reloc = data + data_raw;
    std::memcpy(reloc, &block, sizeof(block));
    std::memcpy(reloc + sizeof(block), entries, sizeof(entries));

    index_rva = CARRIER_DATA_RVA + index_offset;
    return file;
}

constexpr wchar_t CARRIER_PREFIX[] = L"pe_map_tls_";

// The identity of the file behind handle: volume serial number and 128-bit file ID.
bool file_identity(HANDLE handle, FILE_ID_INFO& out) {
    return GetFileInformationByHandleEx(handle, FileIdInfo, &out, sizeof(out)) != FALSE;
}

bool same_file(const FILE_ID_INFO& a, const FILE_ID_INFO& b) {
    return a.VolumeSerialNumber == b.VolumeSerialNumber && std::memcmp(&a.FileId, &b.FileId, sizeof(a.FileId)) == 0;
}

// A protected DACL that grants full access to the process's user only, inherited by the carrier.
bool private_descriptor(PSECURITY_DESCRIPTOR& descriptor) {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    union {
        TOKEN_USER user;
        std::uint8_t bytes[sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE];
    } info;
    DWORD size = 0;
    const bool queried = GetTokenInformation(token, TokenUser, &info, sizeof(info), &size) != FALSE;
    CloseHandle(token);
    wchar_t* sid = nullptr;
    if (!queried || !ConvertSidToStringSidW(info.user.User.Sid, &sid)) return false;
    wchar_t sddl[32 + SECURITY_MAX_SID_SIZE * 4];
    std::swprintf(sddl, sizeof(sddl) / sizeof(sddl[0]), L"D:P(A;OICI;FA;;;%ls)", sid);
    LocalFree(sid);
    return ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &descriptor, nullptr)
        != FALSE;
}

// Writes the carrier into a directory of its own and loads it. The path must keep naming the file
// this call wrote until the loader has opened it:
// - The directory is new, under the temporary directory, with a 128-bit random name and a protected
//   DACL for the process's user only. It is held open without delete sharing, so it cannot be
//   renamed or removed. A reparse point in its place fails the map.
// - The file is created there with CREATE_NEW, written, and identified by its file ID. The write
//   handle has to be closed before the load, because the loader's open does not share writing.
// - It is reopened for reading and deletion, shared for reading only, so from then on no one can
//   write, rename or delete it. The reopened file must have the same ID, and its contents, read
//   back through this handle, must be the bytes written.
// - After the load, the file mapped as the carrier must be the file behind this handle.
// Measured on Windows 10 build 19045: while those two handles are held, opening the file for
// writing, renaming it, deleting it and renaming its directory fail with ERROR_SHARING_VIOLATION,
// renaming the directory above fails with ERROR_ACCESS_DENIED, and LoadLibraryExW still loads it.
// index receives the TLS index Windows assigned to the carrier. Returns null on success or the step
// that failed; error then holds GetLastError. On failure this call deletes what it created and
// still holds a handle to; nothing else is touched. If the carrier cannot be reopened, or the
// directory cannot be queried, the file or directory is left where it is, since without a
// verified handle this call cannot tell it is still its own.
const char* load_carrier(const std::vector<std::uint8_t>& file, DWORD index_rva, TlsCarrier& carrier, DWORD& index,
                         DWORD& error) {
    auto failure = [&](const char* step, DWORD code) {
        unload(carrier);
        error = code;
        return step;
    };
    wchar_t temporary[MAX_PATH + 1], directory[MAX_PATH + 1], path[MAX_PATH + 1];
    DWORD length = GetTempPathW(MAX_PATH + 1, temporary);
    if (length && length <= MAX_PATH) length = GetLongPathNameW(temporary, directory, MAX_PATH + 1);
    if (!length || length > MAX_PATH)
        return failure("cannot find the temporary directory for the TLS carrier", GetLastError());
    std::uint8_t random[16];
    const NTSTATUS generated = BCryptGenRandom(nullptr, random, sizeof(random), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (generated != 0)  // error holds the NTSTATUS here
        return failure("cannot generate a name for the TLS carrier", static_cast<DWORD>(generated));
    wchar_t name[sizeof(random) * 2 + 1];
    for (std::size_t i = 0; i < sizeof(random); ++i) std::swprintf(name + 2 * i, 3, L"%02x", random[i]);
    // The directory name, a separator, the file name and ".dll" must fit.
    constexpr std::size_t prefix = sizeof(CARRIER_PREFIX) / sizeof(CARRIER_PREFIX[0]) - 1;
    if (length + 2 * (prefix + sizeof(random) * 2) + 5 > MAX_PATH)
        return failure("the temporary directory path is too long for the TLS carrier", ERROR_FILENAME_EXCED_RANGE);
    std::swprintf(path, MAX_PATH + 1, L"%ls%ls%ls", directory, CARRIER_PREFIX, name);
    std::wcscpy(directory, path);

    PSECURITY_DESCRIPTOR descriptor = nullptr;
    if (!private_descriptor(descriptor))
        return failure("cannot build the access list of the TLS carrier directory", GetLastError());
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), descriptor, FALSE};
    const BOOL created = CreateDirectoryW(directory, &attributes);
    const DWORD create_error = GetLastError();
    LocalFree(descriptor);
    if (!created) return failure("cannot create the TLS carrier directory", create_error);
    carrier.directory = CreateFileW(directory, DELETE | FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                    nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                                    nullptr);
    if (carrier.directory == INVALID_HANDLE_VALUE)
        return failure("cannot open the TLS carrier directory", GetLastError());
    FILE_BASIC_INFO basic{};
    if (!GetFileInformationByHandleEx(carrier.directory, FileBasicInfo, &basic, sizeof(basic))) {
        const DWORD code = GetLastError();
        CloseHandle(carrier.directory);  // unverified, so not removed
        carrier.directory = INVALID_HANDLE_VALUE;
        return failure("cannot query the TLS carrier directory", code);
    }
    if (!(basic.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) || (basic.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
        CloseHandle(carrier.directory);  // not the directory this call created, so not removed
        carrier.directory = INVALID_HANDLE_VALUE;
        return failure("the TLS carrier directory was replaced", ERROR_SUCCESS);
    }

    std::swprintf(path, MAX_PATH + 1, L"%ls\\%ls%ls.dll", directory, CARRIER_PREFIX, name);
    HANDLE out = CreateFileW(path, GENERIC_WRITE | DELETE, 0, nullptr, CREATE_NEW,
                             FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (out == INVALID_HANDLE_VALUE) return failure("cannot create the TLS carrier file", GetLastError());
    DWORD written = 0;
    FILE_ID_INFO wrote{};
    const bool ok = WriteFile(out, file.data(), static_cast<DWORD>(file.size()), &written, nullptr)
        && written == file.size() && file_identity(out, wrote);
    const DWORD write_error = GetLastError();
    if (!ok) {
        dispose(out);
        CloseHandle(out);
        return failure("cannot write the TLS carrier file", write_error);
    }
    CloseHandle(out);

    // Opened before the load: the loader's own open shares reading and deletion, so this handle can
    // be opened beside it, and it keeps any writer, rename or delete out until the carrier is unloaded.
    HANDLE held = CreateFileW(path, GENERIC_READ | DELETE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (held == INVALID_HANDLE_VALUE) return failure("cannot reopen the TLS carrier file", GetLastError());
    FILE_ID_INFO reopened{};
    if (!file_identity(held, reopened) || !same_file(wrote, reopened)) {
        const DWORD code = GetLastError();
        CloseHandle(held);  // not the file this call wrote, so not deleted
        return failure("the TLS carrier file was replaced before it was loaded", code);
    }
    carrier.file = held;
    std::vector<std::uint8_t> check(file.size() + 1);
    DWORD read = 0;
    if (!ReadFile(carrier.file, check.data(), static_cast<DWORD>(check.size()), &read, nullptr))
        return failure("cannot read back the TLS carrier file", GetLastError());
    if (read != file.size() || std::memcmp(check.data(), file.data(), file.size()) != 0)
        return failure("the TLS carrier file changed before it was loaded", ERROR_SUCCESS);

    carrier.module = LoadLibraryExW(path, nullptr, 0);
    if (!carrier.module) return failure("Windows cannot load the TLS carrier", GetLastError());
    constexpr DWORD NAME = MAX_PATH + 64;
    wchar_t mapped[NAME], held_name[NAME];
    const DWORD mapped_length = GetMappedFileNameW(GetCurrentProcess(), carrier.module, mapped, NAME);
    const DWORD held_length = GetFinalPathNameByHandleW(carrier.file, held_name, NAME, FILE_NAME_OPENED | VOLUME_NAME_NT);
    if (!mapped_length || mapped_length >= NAME || !held_length || held_length >= NAME)
        return failure("cannot name the file mapped as the TLS carrier", GetLastError());
    if (CompareStringOrdinal(mapped, -1, held_name, -1, TRUE) != CSTR_EQUAL)
        return failure("Windows mapped a different file as the TLS carrier", ERROR_SUCCESS);
    std::memcpy(&index, reinterpret_cast<const std::uint8_t*>(carrier.module) + index_rva, sizeof(index));
    if (index == CARRIER_NO_INDEX) return failure("Windows assigned no TLS index to the carrier", ERROR_SUCCESS);
    return nullptr;
}

bool read_file(const wchar_t* path, std::vector<std::uint8_t>& raw) {
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER length{};
    DWORD read = 0;
    bool ok = GetFileSizeEx(file, &length) && length.QuadPart > 0
        && static_cast<std::uint64_t>(length.QuadPart) <= MAX_FILE;
    if (ok) {
        try {
            raw.resize(static_cast<std::size_t>(length.QuadPart));
        } catch (const std::bad_alloc&) {
            ok = false;
        }
    }
    ok = ok && ReadFile(file, raw.data(), static_cast<DWORD>(raw.size()), &read, nullptr) && read == raw.size();
    CloseHandle(file);
    return ok;
}

// Images attached by pe_map_image. pe_map.dll is itself loaded by Windows, so its DllMain receives
// DLL_THREAD_ATTACH/DETACH and forwards them to these images: attach in mapping order, detach in
// reverse mapping order, as measured with two LoadLibraryW copies on Windows 11 build 26300.
//
// Lifetime protocol: each image holds one reference for its registration and one per thread
// notification in flight. DllMain takes those references under mapped_lock and calls the guest
// callbacks after releasing it, so a callback may map or release any image, its own included.
// pe_map_release unregisters the image and drops the registration reference without waiting. The
// image's process detach and unmapping run when the last reference goes, either inside
// pe_map_release or on the thread that finishes the last notification. A notification is admitted
// when its thread takes the reference under the registry lock; none is admitted after the release
// unregisters the image. An admitted notification skips an image already marked released, but one
// whose callback has started finishes: when a TLS callback releases its own image, that image's
// entry point still receives the same thread attach. Addresses are the only identity, so once a
// released address is reused by a new mapping, a stale pointer releases that new mapping.
SRWLOCK mapped_lock = SRWLOCK_INIT;

struct Mapped {
    std::uint8_t* base;
    volatile LONG references;
    volatile LONG released;  // set by pe_map_release; notifiers skip the image from then on
    TlsCarrier carrier;      // holds the image's implicit TLS blocks; empty without a template
};

struct SharedLock {
    SharedLock() { AcquireSRWLockShared(&mapped_lock); }
    ~SharedLock() { ReleaseSRWLockShared(&mapped_lock); }
    SharedLock(const SharedLock&) = delete;
    SharedLock& operator=(const SharedLock&) = delete;
};

struct ExclusiveLock {
    ExclusiveLock() { AcquireSRWLockExclusive(&mapped_lock); }
    ~ExclusiveLock() { ReleaseSRWLockExclusive(&mapped_lock); }
    ExclusiveLock(const ExclusiveLock&) = delete;
    ExclusiveLock& operator=(const ExclusiveLock&) = delete;
};

std::vector<Mapped*>& mapped_images() {
    static std::vector<Mapped*> images;
    return images;
}

// Notify a mapped image, re-reading its callback list as Windows does; if the list was
// overwritten or made unreadable, the entry point is still notified.
void notify_mapped(std::uint8_t* base, DWORD reason) {
    const auto* nt = Image{base, 0}.headers();
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    try {
        if (tls_callbacks(image, nt, callbacks)) callbacks.clear();
    } catch (const std::bad_alloc&) {
        // Never let an allocation failure escape DllMain or pe_map_release; the entry point is
        // still notified.
        callbacks.clear();
    }
    notify(image, nt, callbacks, reason);
}

// Process detach, function-table removal, TLS release and unmapping, once no reference remains.
// The TLS blocks outlive the detach notifications, as they do under FreeLibrary.
void finalize(Mapped* record) {
    std::uint8_t* base = record->base;
    const auto* nt = Image{base, 0}.headers();
    notify_mapped(base, DLL_PROCESS_DETACH);
    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) RtlDeleteFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(base + pdata.VirtualAddress));
    unload(record->carrier);
    VirtualFree(base, 0, MEM_RELEASE);
}

void drop_reference(Mapped* record) {
    if (InterlockedDecrement(&record->references) != 0) return;
    finalize(record);
    delete record;
}

}  // namespace

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH) {
        std::vector<Mapped*> snapshot;
        {
            const SharedLock hold;
            try {
                snapshot.reserve(mapped_images().size());
            } catch (const std::bad_alloc&) {
                return TRUE;  // no references were taken, so nothing is left pending
            }
            for (auto* record : mapped_images()) {
                InterlockedIncrement(&record->references);
                snapshot.push_back(record);
            }
        }
        // Guest callbacks run without mapped_lock. An image released meanwhile is still mapped
        // until its last reference is dropped here.
        auto notify_one = [reason](Mapped* record) {
            if (!InterlockedCompareExchange(&record->released, 0, 0)) notify_mapped(record->base, reason);
            drop_reference(record);
        };
        if (reason == DLL_THREAD_ATTACH) {
            for (auto it = snapshot.begin(); it != snapshot.end(); ++it) notify_one(*it);
        } else {
            for (auto it = snapshot.rbegin(); it != snapshot.rend(); ++it) notify_one(*it);
        }
    }
    return TRUE;
}

extern "C" __declspec(dllexport) void* pe_map_image(const wchar_t* path, Resolver resolve,
                                                    std::uint32_t* status, char* detail,
                                                    std::size_t capacity) {
    const Failure fail{status, detail, capacity};
    if (status) *status = MAP_OK;
    if (detail && capacity) detail[0] = '\0';

    std::vector<std::uint8_t> raw;
    if (!read_file(path, raw)) return fail(MAP_IO, "cannot read image");
    if (raw.size() < sizeof(IMAGE_DOS_HEADER)) return fail(MAP_FORMAT, "truncated DOS header");
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(raw.data());
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0
        || !inside(dos->e_lfanew, sizeof(IMAGE_NT_HEADERS64), raw.size()))
        return fail(MAP_FORMAT, "bad DOS header");
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(raw.data() + dos->e_lfanew);
    const auto& opt = nt->OptionalHeader;
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64
        || opt.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
        return fail(MAP_FORMAT, "not an x64 PE32+ image");
    const auto* sections = IMAGE_FIRST_SECTION(nt);
    const std::uint64_t table_end =
        reinterpret_cast<const std::uint8_t*>(sections + nt->FileHeader.NumberOfSections) - raw.data();
    if (nt->FileHeader.SizeOfOptionalHeader < sizeof(IMAGE_OPTIONAL_HEADER64)
        || dos->e_lfanew + sizeof(IMAGE_NT_HEADERS64) > opt.SizeOfHeaders || table_end > opt.SizeOfHeaders
        || opt.SizeOfHeaders > opt.SizeOfImage || opt.SizeOfHeaders > raw.size())
        return fail(MAP_FORMAT, "section table outside headers");
    // Section protections are applied per page, so a section that started inside a page shared with
    // the headers or another section would change that page's protection as well. Windows refuses an
    // alignment that is not a power of two and a section that does not start on it. An alignment
    // below the page size, which Windows maps as one view when FileAlignment matches, is not modeled.
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    if (opt.SectionAlignment & (opt.SectionAlignment - 1))
        return fail(MAP_FORMAT, "section alignment %#lx is not a power of two", opt.SectionAlignment);
    if (opt.SectionAlignment < system.dwPageSize)
        return fail(MAP_UNSUPPORTED, "section alignment %#lx below the page size", opt.SectionAlignment);
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (sections[i].VirtualAddress & (opt.SectionAlignment - 1))
            return fail(MAP_FORMAT, "section %.8s not aligned to %#lx",
                        reinterpret_cast<const char*>(sections[i].Name), opt.SectionAlignment);
    }

    if (opt.AddressOfEntryPoint && !executable(nt, opt.AddressOfEntryPoint))
        return fail(MAP_FORMAT, "entry point outside executable section");

    // Map away from the preferred base so the relocation path always runs.
    auto allocate = [&] {
        return static_cast<std::uint8_t*>(
            VirtualAlloc(nullptr, opt.SizeOfImage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    };
    auto* base = allocate();
    if (base == reinterpret_cast<std::uint8_t*>(opt.ImageBase)) {
        auto* other = allocate();
        VirtualFree(base, 0, MEM_RELEASE);
        base = other;
    }
    if (!base) return fail(MAP_MEMORY, "cannot reserve image");
    const Image image{base, opt.SizeOfImage};
    TlsCarrier carrier;
    // A carrier that cannot be fully cleaned up is reported after the failure that caused the release.
    auto release = [&](void* result) {
        DWORD error = 0;
        const char* cleanup = unload(carrier, error);
        const std::size_t used = cleanup && detail && capacity ? std::strlen(detail) : capacity;
        if (used + 1 < capacity) std::snprintf(detail + used, capacity - used, "; %s (error %lu)", cleanup, error);
        VirtualFree(base, 0, MEM_RELEASE);
        return result;
    };

    std::memcpy(base, raw.data(), opt.SizeOfHeaders);
    const WORD section_count = nt->FileHeader.NumberOfSections;
    for (WORD i = 0; i < section_count; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        const char* name = reinterpret_cast<const char*>(s.Name);
        if (s.VirtualAddress < opt.SizeOfHeaders || !inside(s.VirtualAddress, span, opt.SizeOfImage))
            return release(fail(MAP_FORMAT, "section %.8s outside image", name));
        // Like the Windows loader, check raw data against the file by two rules only: no raw range
        // may reach 4 GiB, and the last section's raw data must end inside the file. Raw data of an
        // earlier section may lie past the end of the file; the missing bytes stay zero.
        const std::uint64_t raw_end = std::uint64_t{s.PointerToRawData} + s.SizeOfRawData;
        if (s.SizeOfRawData && raw_end > 0xffffffffull)
            return release(fail(MAP_FORMAT, "section %.8s raw data reaches 4 GiB", name));
        if (s.SizeOfRawData && i + 1 == section_count && raw_end > raw.size())
            return release(fail(MAP_FORMAT, "last section %.8s raw data runs past the file", name));
        // A section without a file offset stays zero-filled even when SizeOfRawData is set; it does
        // not take the headers at file offset 0.
        if (!s.PointerToRawData || !s.SizeOfRawData) continue;
        // Read whole 0x200-byte file sectors: round the start of the raw data down and its end up.
        const std::uint64_t first = s.PointerToRawData & ~std::uint64_t{0x1ff};
        std::uint64_t end = (raw_end + 0x1ff) & ~std::uint64_t{0x1ff};
        if (end > raw.size()) end = raw.size();
        if (first >= end) continue;
        const std::uint64_t length = end - first < span ? end - first : span;
        std::memcpy(base + s.VirtualAddress, raw.data() + first, static_cast<std::size_t>(length));
    }

    // A TLS data template is placed after the sections are protected, below. Unlike Windows,
    // which assigns an index even for an empty template, pe_map leaves AddressOfIndex as linked
    // when the template is empty and SizeOfZeroFill is zero.
    // Like the Windows loader, the TLS directory is found by its address alone; Size is not used.
    // Measured on Windows 11 build 26200: a TLS directory with Size 0, 1 or 39 still runs its
    // callback, one with address 0 is ignored whatever its Size, and one that does not fit in the
    // image fails the load.
    const auto& tls = directory(nt, IMAGE_DIRECTORY_ENTRY_TLS);
    if (tls.VirtualAddress) {
        const auto* t = image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress);
        if (!t) return release(fail(MAP_FORMAT, "TLS directory outside image"));
    }

    const auto& imports = directory(nt, IMAGE_DIRECTORY_ENTRY_IMPORT);
    const auto& delay = directory(nt, IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT);

    // Writes below (relocations, IAT) must land in sections, never in the headers or in the
    // directory that drives them, so validated structures cannot change while they are walked.
    auto overlaps = [&](std::uint64_t rva, const IMAGE_DATA_DIRECTORY& driver, std::uint64_t width = 8) {
        const std::uint64_t start = driver.VirtualAddress, end = start + driver.Size;
        return rva + width > start && rva < end;
    };
    auto writable = [&](std::uint64_t rva, const IMAGE_DATA_DIRECTORY& driver, std::uint64_t width = 8) {
        return rva >= opt.SizeOfHeaders && inside(rva, width, image.size) && !overlaps(rva, driver, width);
    };

    // Only complete relocation blocks and complete WORD entries can be applied.
    const auto delta = reinterpret_cast<std::uint64_t>(base) - opt.ImageBase;
    // Like the Windows loader (RtlImageDirectoryEntryToData), a relocation directory at address 0
    // is absent whatever its Size: the image maps as relocation-free instead of the headers being
    // parsed as relocation blocks.
    const auto& linked_relocs = directory(nt, IMAGE_DIRECTORY_ENTRY_BASERELOC);
    static const IMAGE_DATA_DIRECTORY no_relocs{};
    const auto& relocs = linked_relocs.VirtualAddress ? linked_relocs : no_relocs;
    if (nt->FileHeader.Characteristics & IMAGE_FILE_RELOCS_STRIPPED)
        return release(fail(MAP_RELOCATION, "relocations stripped"));
    if (!inside(relocs.VirtualAddress, relocs.Size, opt.SizeOfImage))
        return release(fail(MAP_RELOCATION, "relocation directory outside image"));
    if (relocs.VirtualAddress && relocs.Size) {
        std::uint64_t offset = 0;
        while (offset + sizeof(IMAGE_BASE_RELOCATION) <= relocs.Size) {
            const auto* block = image.at<IMAGE_BASE_RELOCATION>(relocs.VirtualAddress + offset);
            const DWORD page = block->VirtualAddress, length = block->SizeOfBlock;
            if (length < sizeof(IMAGE_BASE_RELOCATION) || length > relocs.Size - offset || length % sizeof(WORD))
                return release(fail(MAP_RELOCATION, "bad relocation block"));
            const auto* entry = reinterpret_cast<const WORD*>(block + 1);
            const DWORD count = (length - sizeof(IMAGE_BASE_RELOCATION)) / sizeof(WORD);
            for (DWORD k = 0; k < count; ++k) {
                const WORD type = entry[k] >> 12;
                const std::uint64_t target = page + std::uint64_t{entry[k] & 0xfffu};
                if (type == IMAGE_REL_BASED_ABSOLUTE) continue;
                const std::uint64_t width = type == IMAGE_REL_BASED_DIR64 ? 8 : type == IMAGE_REL_BASED_HIGHLOW ? 4 : 0;
                if (width == 0 || !writable(target, relocs, width))
                    return release(fail(MAP_RELOCATION, "unsupported relocation at RVA %#llx",
                                        static_cast<unsigned long long>(target)));
                // Entries need not be naturally aligned. Copy only their architectural
                // width: HIGHLOW must neither touch nor carry into the next four bytes.
                if (type == IMAGE_REL_BASED_HIGHLOW) {
                    std::uint32_t value;
                    std::memcpy(&value, image.base + target, sizeof(value));
                    value += static_cast<std::uint32_t>(delta);
                    std::memcpy(image.base + target, &value, sizeof(value));
                } else {
                    std::uint64_t value;
                    std::memcpy(&value, image.base + target, sizeof(value));
                    value += delta;
                    std::memcpy(image.base + target, &value, sizeof(value));
                }
            }
            offset += length;
        }
        if (offset != relocs.Size)
            return release(fail(MAP_RELOCATION, "bad relocation block"));
    }

    // The delay-import directory is found by its address, like Windows; Size is not used. Walk both
    // tables to their null descriptors before either is bound, so an IAT write cannot change the
    // table the other walk reads.
    std::uint64_t delay_descriptors = 0;
    IMAGE_DATA_DIRECTORY delay_walked{};
    if (delay.VirtualAddress) {
        for (;; ++delay_descriptors) {
            const auto* d = image.at<IMAGE_DELAYLOAD_DESCRIPTOR>(
                delay.VirtualAddress + delay_descriptors * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR));
            if (!d) return release(fail(MAP_FORMAT, "delay import directory outside image"));
            if (!d->DllNameRVA) break;
        }
        // The walked table, terminator included, is what IAT writes must not overlap.
        delay_walked = {delay.VirtualAddress,
                        static_cast<DWORD>((delay_descriptors + 1) * sizeof(IMAGE_DELAYLOAD_DESCRIPTOR))};
    }

    // Like the Windows loader, find the import table by its address and walk it to the null
    // descriptor; the directory Size is not used. Measured on Windows 11 builds 26200 and 26300:
    // copies with Size 0, 19 or 20 still load and bind every slot. Each descriptor is
    // bounds-checked on its own.
    std::uint64_t import_descriptors = 0;
    IMAGE_DATA_DIRECTORY imports_walked{};
    if (imports.VirtualAddress) {
        for (;; ++import_descriptors) {
            const auto* d = image.at<IMAGE_IMPORT_DESCRIPTOR>(
                imports.VirtualAddress + import_descriptors * sizeof(IMAGE_IMPORT_DESCRIPTOR));
            if (!d) return release(fail(MAP_FORMAT, "import directory outside image"));
            if (!d->Name) break;
        }
        // The walked table, terminator included, is what IAT writes must not overlap.
        imports_walked = {imports.VirtualAddress,
                          static_cast<DWORD>((import_descriptors + 1) * sizeof(IMAGE_IMPORT_DESCRIPTOR))};
    }

    // The PE loader leaves delay imports to the image's helper, which calls LoadLibrary and would
    // bypass the host resolver. pe_map has no LoadLibrary, so it resolves each delay descriptor
    // through the same resolver at map time and writes the bound address into its IAT. A descriptor
    // without dlattrRva stores virtual addresses rather than RVAs and is rejected. The directory is
    // found by its address, like Windows; Size is not used, and the table is walked above.
    if (delay_descriptors) {
        if (!resolve) return release(fail(MAP_UNRESOLVED_IMPORT, "delay imports present without a resolver"));
        const auto* desc = image.at<IMAGE_DELAYLOAD_DESCRIPTOR>(delay.VirtualAddress, delay_descriptors + 1);
        if (!desc) return release(fail(MAP_FORMAT, "delay import directory outside image"));
        for (std::uint64_t d = 0; d < delay_descriptors; ++d) {
            if (!(desc[d].Attributes.AllAttributes & DELAY_IMPORT_RVA_BASED))
                return release(fail(MAP_UNSUPPORTED, "delay imports without RVA attributes"));
            const char* name_in_image = image.string(desc[d].DllNameRVA);
            if (!name_in_image) return release(fail(MAP_FORMAT, "delay import DLL name outside image"));
            // Keep a copy: an IAT write below may legally overlap the string in the image.
            const std::string dll_name(name_in_image);
            const char* dll = dll_name.c_str();
            const DWORD lookup = desc[d].ImportNameTableRVA, iat = desc[d].ImportAddressTableRVA;
            if (!lookup || !iat)
                return release(fail(MAP_FORMAT, "delay import tables for %s outside image", dll));
            for (std::uint64_t i = 0;; ++i) {
                const auto* word = image.at<std::uint64_t>(lookup + 8 * i);
                if (!word) return release(fail(MAP_FORMAT, "delay import thunk for %s outside image", dll));
                const std::uint64_t slot = iat + 8 * i;
                if (!writable(slot, delay_walked) || overlaps(slot, imports_walked))
                    return release(fail(MAP_FORMAT, "delay import IAT for %s outside image", dll));
                if (!*word) break;
                const bool by_ordinal = IMAGE_SNAP_BY_ORDINAL64(*word);
                const auto ordinal = by_ordinal ? static_cast<std::uint32_t>(IMAGE_ORDINAL64(*word)) : 0u;
                // Bits 31..62 of a named thunk must be zero: masking them could alias a valid low RVA.
                const std::uint64_t name_rva = by_ordinal ? 0 : *word;
                const auto* hint = by_ordinal || name_rva >> 31 ? nullptr : image.at<WORD>(name_rva);
                const char* name = hint ? image.string(name_rva + sizeof(WORD)) : nullptr;
                if (!by_ordinal && !name)
                    return release(fail(MAP_FORMAT, "delay import name for %s outside image", dll));
                void* bound = resolve(dll, name, ordinal);
                if (!bound && name)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved delay import %s!%s", dll, name));
                if (!bound)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved delay import %s!#%u", dll,
                                        static_cast<unsigned>(ordinal)));
                *image.at<std::uint64_t>(slot) = reinterpret_cast<std::uint64_t>(bound);
            }
        }
    }

    if (import_descriptors) {
        if (!resolve) return release(fail(MAP_UNRESOLVED_IMPORT, "imports present without a resolver"));
        const auto* desc = image.at<IMAGE_IMPORT_DESCRIPTOR>(imports.VirtualAddress, import_descriptors + 1);
        for (std::uint64_t d = 0; d < import_descriptors; ++d) {
            const char* name_in_image = image.string(desc[d].Name);
            if (!name_in_image) return release(fail(MAP_FORMAT, "import DLL name outside image"));
            // Keep a copy: an IAT write below may legally overlap the string in the image.
            const std::string dll_name(name_in_image);
            const char* dll = dll_name.c_str();
            const DWORD lookup = desc[d].OriginalFirstThunk ? desc[d].OriginalFirstThunk : desc[d].FirstThunk;
            for (std::uint64_t i = 0;; ++i) {
                const auto* word = image.at<std::uint64_t>(lookup + 8 * i);
                const std::uint64_t target = desc[d].FirstThunk + 8 * i;
                // The delay directory was read above, but a normal IAT write must still not reach it.
                if (!word || !writable(target, imports_walked) || overlaps(target, delay_walked))
                    return release(fail(MAP_FORMAT, "import thunk for %s outside image", dll));
                if (!*word) break;
                const bool by_ordinal = IMAGE_SNAP_BY_ORDINAL64(*word);
                const auto ordinal = by_ordinal ? static_cast<std::uint32_t>(IMAGE_ORDINAL64(*word)) : 0u;
                const auto* hint = by_ordinal ? nullptr : image.at<WORD>(*word & 0x7fffffff);
                const char* name = hint ? image.string((*word & 0x7fffffff) + sizeof(WORD)) : nullptr;
                if (!by_ordinal && !name) return release(fail(MAP_FORMAT, "import name for %s outside image", dll));
                void* bound = resolve(dll, name, ordinal);
                if (!bound && name) return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!%s", dll, name));
                if (!bound)
                    return release(fail(MAP_UNRESOLVED_IMPORT, "unresolved import %s!#%u", dll,
                                        static_cast<unsigned>(ordinal)));
                *image.at<std::uint64_t>(target) = reinterpret_cast<std::uint64_t>(bound);
            }
        }
    }

    // Record the actual base in the mapped headers, as the Windows loader does.
    reinterpret_cast<IMAGE_NT_HEADERS64*>(base + dos->e_lfanew)->OptionalHeader.ImageBase =
        reinterpret_cast<std::uint64_t>(base);
    DWORD old = 0;
    if (!VirtualProtect(base, opt.SizeOfHeaders, PAGE_READONLY, &old))
        return release(fail(MAP_PROTECT, "cannot protect headers"));
    for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        const auto& s = sections[i];
        const DWORD span = s.Misc.VirtualSize ? s.Misc.VirtualSize : s.SizeOfRawData;
        if (span && !VirtualProtect(base + s.VirtualAddress, span, protection(s.Characteristics), &old))
            return release(fail(MAP_PROTECT, "cannot protect section %.8s", reinterpret_cast<const char*>(s.Name)));
    }
    FlushInstructionCache(GetCurrentProcess(), base, opt.SizeOfImage);

    // An empty exception directory registers nothing. Require complete entries otherwise.
    const auto& pdata = directory(nt, IMAGE_DIRECTORY_ENTRY_EXCEPTION);
    if (pdata.Size) {
        if (!pdata.VirtualAddress || pdata.Size % sizeof(RUNTIME_FUNCTION))
            return release(fail(MAP_FUNCTION_TABLE, "cannot register .pdata"));
        const auto count = pdata.Size / sizeof(RUNTIME_FUNCTION);
        auto* table = image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress, count);
        if (table && !readable(table, pdata.Size))
            return release(fail(MAP_UNSUPPORTED, "exception table is not readable after section protection"));
        if (!table || !RtlAddFunctionTable(table, static_cast<DWORD>(count),
                                           reinterpret_cast<DWORD64>(base)))
            return release(fail(MAP_FUNCTION_TABLE, "cannot register .pdata"));
    }
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    const auto* mapped_nt = image.headers();
    if (const char* reason = tls_callbacks(image, mapped_nt, callbacks)) {
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        const bool limit = reason == TOO_MANY_TLS_CALLBACKS;
        return release(fail(limit ? MAP_UNSUPPORTED : MAP_FORMAT, "%s", reason));
    }
    // Implicit TLS data, before any guest code runs. tls_callbacks() has validated the directory
    // and AddressOfIndex. Measured on Windows 10 build 19045 with tests/cpu/tls_data.cpp: a template
    // that ends past the image fails the load with ERROR_NOACCESS and one that ends before its
    // start with ERROR_BAD_EXE_FORMAT; a non-zero SizeOfZeroFill and every Characteristics value
    // tried still load, so both fields are handed to Windows unchanged in the carrier.
    if (tls.VirtualAddress) {
        auto fail_tls = [&](MapStatus code, const char* reason) {
            if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
            return release(fail(code, "%s", reason));
        };
        const auto* t = image.at<IMAGE_TLS_DIRECTORY64>(tls.VirtualAddress);
        const auto address = reinterpret_cast<std::uint64_t>(base);
        const std::uint64_t start = t->StartAddressOfRawData, end = t->EndAddressOfRawData;
        const DWORD zero_fill = t->SizeOfZeroFill, characteristics = t->Characteristics;
        const std::uint64_t index_va = t->AddressOfIndex;
        if (end != start || zero_fill) {
            if (end < start) return fail_tls(MAP_FORMAT, "TLS template ends before it starts");
            // An empty template names no bytes, so its address is not checked: the linker may leave it 0.
            if (end != start && (start < address || !inside(start - address, end - start, image.size)))
                return fail_tls(MAP_FORMAT, "TLS template outside image");
            if (end - start > MAX_TLS_TEMPLATE || zero_fill > MAX_TLS_TEMPLATE)
                return fail_tls(MAP_UNSUPPORTED, "TLS template larger than 16 MiB");
            const std::uint8_t* bytes = end != start ? base + (start - address) : base;
            const auto length = static_cast<std::size_t>(end - start);
            if (!readable(bytes, length)) return fail_tls(MAP_UNSUPPORTED, "TLS template not readable");
            // The slot lies inside a writable section of the image, see tls_callbacks().
            auto* slot = reinterpret_cast<std::uint8_t*>(index_va);
            DWORD index_rva = 0, index = 0, error = 0;
            const char* failed = nullptr;
            try {
                const auto file = carrier_image(bytes, length, zero_fill, characteristics, index_rva);
                failed = load_carrier(file, index_rva, carrier, index, error);
            } catch (const std::bad_alloc&) {
                return fail_tls(MAP_MEMORY, "cannot allocate the TLS carrier");
            }
            if (failed) {
                if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
                return release(fail(MAP_MEMORY, "%s (error %lu)", failed, error));
            }
            std::memcpy(slot, &index, sizeof(index));
        }
    }
    if (!notify(image, mapped_nt, callbacks, DLL_PROCESS_ATTACH)) {
        // Measured on Windows: a failed attach is followed by a full detach before the image is freed.
        // The callback list collected before attach is reused here; only pe_map_release re-reads it.
        notify(image, mapped_nt, callbacks, DLL_PROCESS_DETACH);
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        return release(fail(MAP_INIT_FAILED, "entry point returned FALSE"));
    }
    Mapped* record = new (std::nothrow) Mapped{base, 1, 0, carrier};
    bool registered = record != nullptr;
    if (record) {
        const ExclusiveLock hold;
        try {
            mapped_images().push_back(record);
        } catch (const std::bad_alloc&) {
            registered = false;
        }
    }
    if (!registered) {
        delete record;
        notify(image, mapped_nt, callbacks, DLL_PROCESS_DETACH);
        if (pdata.Size) RtlDeleteFunctionTable(image.at<RUNTIME_FUNCTION>(pdata.VirtualAddress));
        return release(fail(MAP_MEMORY, "cannot register image for thread notifications"));
    }
    carrier = TlsCarrier{};  // owned by the record from here on
    return base;
}

namespace {

struct ExportLookup {
    DWORD rva = 0;
    const char* forwarder = nullptr;
};

// Finds one named export. The export directory, its three tables and the forwarder string are
// checked against the live page protection, as the TLS path does.
bool find_export(const Image& image, const IMAGE_NT_HEADERS64* nt, const char* name, ExportLookup& out) {
    const auto& exports = directory(nt, IMAGE_DIRECTORY_ENTRY_EXPORT);
    if (exports.Size < sizeof(IMAGE_EXPORT_DIRECTORY) || !inside(exports.VirtualAddress, exports.Size, image.size))
        return false;
    const auto* dir = exports.Size ? image.at<IMAGE_EXPORT_DIRECTORY>(exports.VirtualAddress) : nullptr;
    if (!dir || !readable(dir, sizeof(IMAGE_EXPORT_DIRECTORY))) return false;
    const auto* names = image.at<DWORD>(dir->AddressOfNames, dir->NumberOfNames);
    const auto* ordinals = image.at<WORD>(dir->AddressOfNameOrdinals, dir->NumberOfNames);
    const auto* functions = image.at<DWORD>(dir->AddressOfFunctions, dir->NumberOfFunctions);
    if (!names || !ordinals || !functions) return false;
    if (!readable(names, sizeof(DWORD) * dir->NumberOfNames) ||
        !readable(ordinals, sizeof(WORD) * dir->NumberOfNames) ||
        !readable(functions, sizeof(DWORD) * dir->NumberOfFunctions)) return false;
    for (DWORD i = 0; i < dir->NumberOfNames; ++i) {
        const char* exported = readable_string(image, names[i]);
        if (!exported || std::strcmp(exported, name) != 0) continue;
        if (ordinals[i] >= dir->NumberOfFunctions) return false;
        const DWORD rva = functions[ordinals[i]];
        // A forwarded export points back into the export directory, where its target string lives.
        if (rva >= exports.VirtualAddress && rva - exports.VirtualAddress < exports.Size) {
            const Image directory_view{image.base, std::uint64_t{exports.VirtualAddress} + exports.Size};
            out.forwarder = readable_string(directory_view, rva);
            return out.forwarder != nullptr;
        }
        out.rva = rva;
        return true;
    }
    return false;
}

}  // namespace

// Only valid for a base returned by pe_map_image, whose headers were validated and are read-only.
extern "C" __declspec(dllexport) void* pe_map_export(void* mapped, const char* name) {
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    ExportLookup found;
    // A forwarded export is not code in this image, so this entry point keeps returning null.
    if (!find_export(image, nt, name, found) || found.forwarder) return nullptr;
    // An empty slot (RVA 0) names no code; returning the image base would be wrong. An exported
    // address must be accessible, but it may be execute-only, so readable() is not used here.
    return found.rva && found.rva < image.size && accessible(image.base + found.rva)
        ? image.base + found.rva : nullptr;
}

// Like pe_map_export, but a forwarded export is resolved through the host resolver. The forwarder
// string is "Module.Symbol" or "Module.#ordinal", with the module as the linker recorded it.
extern "C" __declspec(dllexport) void* pe_map_export_resolved(void* mapped, const char* name, Resolver resolve) {
    const auto* nt = Image{static_cast<std::uint8_t*>(mapped), 0}.headers();
    const Image image{static_cast<std::uint8_t*>(mapped), nt->OptionalHeader.SizeOfImage};
    ExportLookup found;
    if (!find_export(image, nt, name, found)) return nullptr;
    if (!found.forwarder)
        return found.rva && found.rva < image.size && accessible(image.base + found.rva)
            ? image.base + found.rva : nullptr;
    if (!resolve) return nullptr;
    const std::string target(found.forwarder);
    const auto dot = target.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 1 == target.size()) return nullptr;
    const std::string module = target.substr(0, dot);
    const std::string symbol = target.substr(dot + 1);
    if (symbol.front() == '#') {
        std::uint32_t ordinal = 0;
        for (std::size_t i = 1; i < symbol.size(); ++i) {
            const char digit = symbol[i];
            if (digit < '0' || digit > '9') return nullptr;
            ordinal = ordinal * 10 + static_cast<unsigned>(digit - '0');
            if (ordinal > 65535) return nullptr;
        }
        return ordinal ? resolve(module.c_str(), nullptr, ordinal) : nullptr;
    }
    return resolve(module.c_str(), symbol.c_str(), 0);
}

// Returns 1 after unregistering an image that pe_map_image returned, and 0 for null, for any
// other pointer and for a second release. Process detach and unmapping follow the protocol above:
// at once if no thread notification for the image is in flight, otherwise when the last one ends.
extern "C" __declspec(dllexport) int pe_map_release(void* mapped) {
    Mapped* record = nullptr;
    {
        const ExclusiveLock hold;
        auto& images = mapped_images();
        for (auto it = images.begin(); it != images.end(); ++it) {
            if ((*it)->base == mapped) {
                record = *it;
                images.erase(it);
                break;
            }
        }
    }
    if (!record) return 0;
    InterlockedExchange(&record->released, 1);
    drop_reference(record);
    return 1;
}

// Start a mapped EXE on the calling thread the way the loader starts a process: TLS callbacks in table
// order with DLL_PROCESS_ATTACH and a null reserved argument, then the entry point, whose return value is
// the exit code Windows would use. Measured on Windows 11 build 26300 with tests/cpu/exe_start.asm.
//
// Preconditions, none of which is checked:
// - `mapped` is a base returned by pe_map_image, still live and not yet released. Like pe_map_export,
//   this reads the headers of the argument directly, so any other pointer is undefined behavior.
// - The caller trusts the image. The entry point and the callbacks are arbitrary machine code that runs
//   in the host process with the host's privileges. They can crash, hang or terminate it (an entry point
//   that calls ExitProcess ends the host), and they run on the calling thread, so their stack, TLS and
//   loader-lock state are the caller's. Run untrusted or unknown images in a child process.
// pe_map_image maps an EXE without notifying it, like LoadLibrary, so nothing has run before this call.
// Start once per mapping. The caller keeps it live throughout the call and prevents any thread,
// TLS callback or entry point from re-entering run or releasing this mapping during startup.
//
// Returns 0 without running anything for a DLL, which pe_map_image already notified, for an image without
// an entry point, for a null exit_code and for an unreadable TLS callback list. The PEB argument that
// Windows passes to the entry point is not forwarded, and no process-detach notification follows because
// the process goes on.
extern "C" __declspec(dllexport) int pe_map_run(void* mapped, std::uint32_t* exit_code) {
    auto* base = static_cast<std::uint8_t*>(mapped);
    const auto* nt = Image{base, 0}.headers();
    if (!exit_code || (nt->FileHeader.Characteristics & IMAGE_FILE_DLL) || !nt->OptionalHeader.AddressOfEntryPoint)
        return 0;
    const Image image{base, nt->OptionalHeader.SizeOfImage};
    std::vector<PIMAGE_TLS_CALLBACK> callbacks;
    if (tls_callbacks(image, nt, callbacks)) return 0;
    for (auto callback : callbacks) callback(base, DLL_PROCESS_ATTACH, nullptr);
    using Entry = DWORD(WINAPI*)(void*);
    *exit_code = reinterpret_cast<Entry>(base + nt->OptionalHeader.AddressOfEntryPoint)(nullptr);
    return 1;
}
