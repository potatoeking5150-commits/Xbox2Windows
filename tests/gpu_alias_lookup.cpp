// Four title allocations (three era_heap graphics allocations and one placed across a 4 GiB line)
// opened as D3D12 heaps (era_gpu_alias) and packed for the shader lookup
// (era_title_va::shader_ranges, src/title_va_lookup.hlsli). Compute shaders take 64-bit title
// addresses from an input buffer and load (check_gpu_alias_lookup) or store and update with
// atomics (check_gpu_alias_lookup_writes) through them. Expected values come from the CPU
// through the title pointers.
#include "bvh_fixture.h"
#include "era_gpu_alias.h"
#include "era_heap.h"
#include "era_title_va.h"
#include "LookupTitleAddresses.h"
#include "ProbeArrayAtomics.h"
#include "ProbeArrayAtomicsWaterfall.h"
#include "ProbeArrayAtomicsReturns.h"
#include "WriteTitleAddresses.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace bvh_fixture;

namespace {
// XALLOC_ATTRIBUTES: memoryType 1 (graphics), pageSize 1 (64 KB), alignment 16 (64 KB).
constexpr std::uint64_t graphics_64k = (1ull << 29) | (1ull << 14) | (16ull << 24);
constexpr std::array<std::size_t, 3> sizes{256 * 1024, 64 * 1024, 128 * 1024};
constexpr std::size_t straddle_bytes = 128 * 1024;
constexpr UINT descriptor_count = 8;  // matches Aliases[8] in the shaders
constexpr Word miss = 0xdeadbeef;

struct TitleAllocation {
    explicit TitleAllocation(std::size_t bytes) : base(era_heap::allocate(bytes, graphics_64k)), size(bytes) {}
    // An allocation placed across a 4 GiB line, so the lookup has to compare and subtract the
    // high 32-bit words correctly. Tries lines from 8 GiB up; base stays null if none is free.
    TitleAllocation() : base(nullptr), size(straddle_bytes), placed(true)
    {
        for (std::uint64_t line = 2; line < 128 && !base; ++line)
            base = VirtualAlloc(reinterpret_cast<void*>((line << 32) - straddle_bytes / 2), straddle_bytes,
                                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
    ~TitleAllocation()
    {
        if (placed) {
            if (base) VirtualFree(base, 0, MEM_RELEASE);
        } else {
            era_heap::release_with_attributes(base, graphics_64k);
        }
    }
    TitleAllocation(const TitleAllocation&) = delete;
    TitleAllocation& operator=(const TitleAllocation&) = delete;
    std::uint64_t begin() const { return reinterpret_cast<std::uint64_t>(base); }
    // The placed allocation's range stops two bytes short of the allocation, so its last
    // aligned word crosses the range end while its first two bytes are still inside.
    std::size_t range_bytes() const { return placed ? size - 2 : size; }
    // Start of the last aligned DWORD wholly inside the range.
    std::uint64_t last_word() const { return begin() + ((range_bytes() - 4) & ~std::size_t{3}); }
    void* base;
    std::size_t size;
    bool placed = false;
};

Word pattern(std::size_t allocation, std::size_t offset)
{
    return static_cast<Word>((allocation + 1) * 0x01000193u ^ (offset * 2654435761u));
}

Word& word_at(std::uint64_t address) { return *reinterpret_cast<Word*>(address); }

// The allocations, their aliases and table, the packed ranges and a descriptor table whose slot
// i holds a raw UAV of the alias of ranges[i].
struct AliasFixture {
    std::vector<std::unique_ptr<TitleAllocation>> titles;
    std::vector<era_gpu_alias::Alias> aliases;
    era_title_va::Table table;
    std::vector<era_title_va::Range> ranges;
    std::vector<era_title_va::ShaderRange> packed;
    ComPtr<ID3D12DescriptorHeap> heap;

    // Returns false when the adapter does not support existing heaps.
    bool open(ID3D12Device* device)
    {
        for (const auto bytes : sizes) titles.push_back(std::make_unique<TitleAllocation>(bytes));
        titles.push_back(std::make_unique<TitleAllocation>());
        aliases.resize(titles.size());
        for (std::size_t a = 0; a < titles.size(); ++a) {
            auto* base = static_cast<unsigned char*>(titles[a]->base);
            if (!base) throw std::runtime_error("title allocation failed");
            const HRESULT hr = era_gpu_alias::open(device, base, titles[a]->size, aliases[a]);
            if (hr == E_NOTIMPL) {
                std::puts("existing heaps are not supported by this adapter");
                return false;
            }
            check(hr, "Open a title allocation as a D3D12 heap");
            if (!table.add(titles[a]->begin(), titles[a]->range_bytes(), aliases[a].buffer->GetGPUVirtualAddress()))
                throw std::runtime_error("table rejected an alias");
            for (std::size_t offset = 0; offset < titles[a]->size; offset += sizeof(Word))
                *reinterpret_cast<Word*>(base + offset) = pattern(a, offset);
        }
        ranges = table.ranges();
        if (!era_title_va::shader_ranges(ranges, packed)) throw std::runtime_error("shader_ranges rejected the table");
        if ((line() & 0xffffffffull) != 0) throw std::runtime_error("placed allocation does not straddle a 4 GiB line");

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptor_count,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create lookup descriptor heap");
        const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (UINT slot = 0; slot < descriptor_count; ++slot) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_R32_TYPELESS;
            view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            ID3D12Resource* resource = nullptr;
            if (slot < ranges.size()) {
                const auto a = allocation_of(ranges[slot].title_begin);
                resource = aliases[a].buffer.Get();
                view.Buffer.NumElements = static_cast<UINT>(titles[a]->size / sizeof(Word));
            }
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
            cpu.ptr += static_cast<SIZE_T>(slot) * stride;
            device->CreateUnorderedAccessView(resource, nullptr, &view, cpu);
        }
        return true;
    }

    std::size_t allocation_of(std::uint64_t title_begin) const
    {
        for (std::size_t a = 0; a < titles.size(); ++a)
            if (titles[a]->begin() == title_begin) return a;
        throw std::runtime_error("range without an allocation");
    }

    // The 4 GiB line inside the placed allocation.
    std::uint64_t line() const { return titles.back()->begin() + straddle_bytes / 2; }

    // Slot + 1 of the range holding the aligned DWORD [address, address + 4), or 0.
    Word slot_of(std::uint64_t address) const
    {
        if ((address & 3) != 0 || !table.translate(address, sizeof(Word))) return 0;
        for (std::size_t slot = 0; slot < ranges.size(); ++slot)
            if (address >= ranges[slot].title_begin && address < ranges[slot].title_end)
                return static_cast<Word>(slot + 1);
        return 0;
    }

    // Runs `kernel` once per record and returns the results buffer's words.
    template<class Record>
    std::vector<Word> run(ID3D12Device* device, const Kernel& kernel, const std::vector<Record>& records,
                          std::size_t result_words)
    {
        auto range_buffer = upload(device, packed);
        auto record_buffer = upload(device, records);
        const std::size_t result_bytes = result_words * sizeof(Word);
        auto results = make_buffer(device, result_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto back = make_buffer(device, result_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        ComputeCommands cmd(device, kernel.pipeline.Get());
        auto* list = cmd.list.Get();
        for (auto& alias : aliases)
            transition(list, alias.buffer.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(list, results.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(kernel.root.Get());
        list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        list->SetComputeRootShaderResourceView(1, range_buffer->GetGPUVirtualAddress());
        list->SetComputeRootShaderResourceView(2, record_buffer->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(3, results->GetGPUVirtualAddress());
        const auto record_count = static_cast<UINT>(records.size());
        list->SetComputeRoot32BitConstant(4, static_cast<UINT>(packed.size()), 0);
        list->SetComputeRoot32BitConstant(4, record_count, 1);
        list->Dispatch((record_count + 63) / 64, 1, 1);
        for (auto& alias : aliases)
            transition(list, alias.buffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        transition(list, results.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(back.Get(), 0, results.Get(), 0, result_bytes);
        cmd.submit_and_wait(device);
        return read(back.Get(), result_words);
    }
};

Word float_bits(float value)
{
    Word bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Whether InterlockedAdd through a per-thread index into a UAV array updates each word exactly
// once per thread, measured on plain committed buffers so the answer does not depend on the
// aliases under test. 256 threads each add 1 to word 0 of one of four zeroed buffers, 64 threads
// per buffer. Prints the expected and actual counts. The in-box WARP of Windows 10 (10.0.19041)
// applies each such add four times, which is a wrong result, not a missing capability.
bool array_atomics_exact(ID3D12Device* device, D3D12_SHADER_BYTECODE shader, const char* name)
{
    constexpr UINT targets = 4, threads = 256;
    Kernel kernel(device, shader);
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptor_count,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "Create atomics probe descriptor heap");
    const UINT stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    std::array<ComPtr<ID3D12Resource>, targets> buffers;
    for (UINT slot = 0; slot < descriptor_count; ++slot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        view.Buffer.NumElements = 1;
        ID3D12Resource* resource = nullptr;
        if (slot < targets) {
            buffers[slot] = make_buffer(device, sizeof(Word), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            resource = buffers[slot].Get();
        }
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(slot) * stride;
        device->CreateUnorderedAccessView(resource, nullptr, &view, cpu);
    }
    auto zero = upload(device, std::vector<Word>(1, 0));
    auto back = make_buffer(device, targets * sizeof(Word), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands cmd(device, kernel.pipeline.Get());
    auto* list = cmd.list.Get();
    for (auto& buffer : buffers) {
        list->CopyBufferRegion(buffer.Get(), 0, zero.Get(), 0, sizeof(Word));
        transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(kernel.root.Get());
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRoot32BitConstant(1, targets, 0);
    list->SetComputeRoot32BitConstant(1, threads, 1);
    list->Dispatch(threads / 64, 1, 1);
    for (UINT i = 0; i < targets; ++i) {
        transition(list, buffers[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(back.Get(), i * sizeof(Word), buffers[i].Get(), 0, sizeof(Word));
    }
    cmd.submit_and_wait(device);
    const auto counts = read(back.Get(), targets);
    const Word expected = threads / targets;
    std::printf("%s threads=%u buffers=%u expected=%u actual=%u,%u,%u,%u\n", name, threads, targets, expected,
                counts[0], counts[1], counts[2], counts[3]);
    return std::all_of(counts.begin(), counts.end(), [=](Word count) { return count == expected; });
}
}
bool check_gpu_alias_atomic_waterfall(ID3D12Device* device)
{
    return array_atomics_exact(device, {ProbeArrayAtomicsWaterfall, sizeof(ProbeArrayAtomicsWaterfall)},
                               "array_atomics_waterfall");
}
bool check_gpu_alias_atomic_returns(ID3D12Device* device)
{
    constexpr UINT targets = 4, threads = 256;
    Kernel kernel(device, {ProbeArrayAtomicsReturns, sizeof(ProbeArrayAtomicsReturns)});

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, descriptor_count,
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0
    };
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)),
          "Create atomic return probe descriptor heap");

    const UINT stride =
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    std::array<ComPtr<ID3D12Resource>, targets> buffers;
    auto results = make_buffer(device, threads * sizeof(Word),
                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    for (UINT i = 0; i < targets; ++i)
        buffers[i] = make_buffer(device, sizeof(Word), D3D12_HEAP_TYPE_DEFAULT,
                                 D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    for (UINT slot = 0; slot < descriptor_count; ++slot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        view.Buffer.NumElements = slot == 4 ? threads : 1;

        ID3D12Resource* resource = nullptr;
        if (slot < targets) resource = buffers[slot].Get();
        else if (slot == 4) resource = results.Get();

        D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(slot) * stride;
        device->CreateUnorderedAccessView(resource, nullptr, &view, cpu);
    }

    auto zero = upload(device, std::vector<Word>(1, 0));
    auto count_back = make_buffer(device, targets * sizeof(Word),
                                  D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    auto values_back = make_buffer(device, threads * sizeof(Word),
                                   D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    ComputeCommands cmd(device, kernel.pipeline.Get());
    auto* list = cmd.list.Get();

    for (auto& buffer : buffers) {
    transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(buffer.Get(), 0, zero.Get(), 0, sizeof(Word));
    transition(list, buffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}
    transition(list, results.Get(), D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(kernel.root.Get());
    list->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    list->SetComputeRoot32BitConstant(1, targets, 0);
    list->SetComputeRoot32BitConstant(1, threads, 1);
    list->Dispatch(threads / 64, 1, 1);

    for (UINT i = 0; i < targets; ++i) {
        transition(list, buffers[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(count_back.Get(), i * sizeof(Word),
                               buffers[i].Get(), 0, sizeof(Word));
    }
    transition(list, results.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(values_back.Get(), 0, results.Get(), 0,
                           threads * sizeof(Word));

    cmd.submit_and_wait(device);

    const auto counts = read(count_back.Get(), targets);
    const auto values = read(values_back.Get(), threads);

    std::array<std::array<bool, 64>, targets> seen{};
    bool valid = true;

    for (UINT i = 0; i < targets; ++i)
        valid = valid && counts[i] == 64;

    for (UINT id = 0; id < threads; ++id) {
        const UINT buffer = id % targets;
        const Word value = values[id];

        if (value >= 64 || seen[buffer][value]) {
            valid = false;
            continue;
        }
        seen[buffer][value] = true;
    }

    for (const auto& buffer_seen : seen)
        for (bool found : buffer_seen)
            valid = valid && found;

    std::printf("array_atomics_returns counters=%u,%u,%u,%u return_values=%s\n",
                counts[0], counts[1], counts[2], counts[3],
                valid ? "PASS" : "FAIL");

    return valid;
}
bool check_gpu_alias_lookup(ID3D12Device* device)
{
    AliasFixture fixture;
    if (!fixture.open(device)) return false;

    // Records: start, second word, middle words, last word and the words just outside each
    // allocation (outside words count as hits only if another allocation happens to be there),
    // plus the null page and an address with a high 32-bit word above any user address.
    std::vector<std::uint64_t> records;
    for (const auto& title : fixture.titles) {
        const auto begin = title->begin();
        const auto end = begin + title->range_bytes();
        for (std::uint64_t offset : {std::uint64_t{0}, std::uint64_t{4}, title->size / 2, title->size / 2 + 0x124})
            records.push_back(begin + offset);
        // The last valid word, the aligned word after it (outside the range, or crossing its end
        // for the placed allocation), and misaligned starts at both ends.
        records.push_back(title->last_word());
        records.push_back(title->last_word() + 4);
        for (std::uint64_t address : {begin + 1, begin + 2, begin + 3, end - 1, end - 2, end - 3})
            records.push_back(address);
        records.push_back(begin - 4);
        records.push_back(end);
        for (std::uint64_t offset = 0; offset < title->size; offset += 0x1f04) records.push_back(begin + offset);
    }
    // Around the 4 GiB line inside the placed allocation, and the same low words one 4 GiB
    // above and below, which only a full 64-bit comparison rejects.
    const auto line = fixture.line();
    for (std::uint64_t address : {line - 8, line - 4, line, line + 4, line + 0x1234}) {
        records.push_back(address);
        records.push_back(address + (1ull << 32));
        records.push_back(address - (1ull << 32));
    }
    records.push_back(0x10);
    records.push_back(0xfffffffffffffff0ull);

    // Expected results from the CPU through the title pointers.
    std::vector<Word> expected(records.size() * 2);
    Word hits = 0, misses = 0;
    for (std::size_t r = 0; r < records.size(); ++r) {
        const Word slot = fixture.slot_of(records[r]);
        expected[r * 2] = slot ? word_at(records[r]) : miss;
        expected[r * 2 + 1] = slot;
        ++(slot ? hits : misses);
    }
    // 12 out-of-range addresses, 24 misaligned starts and the placed range's tail-crossing word.
    if (misses < 37) throw std::runtime_error("fixture needs the 37 guaranteed misses");

    Kernel kernel(device, {LookupTitleAddresses, sizeof(LookupTitleAddresses)});
    const auto actual = fixture.run(device, kernel, records, expected.size());
    for (std::size_t r = 0; r < records.size(); ++r) {
        equal(actual[r * 2], expected[r * 2], static_cast<unsigned>(r), 0);
        equal(actual[r * 2 + 1], expected[r * 2 + 1], static_cast<unsigned>(r), 1);
    }
    std::printf("gpu_alias_lookup ranges=%zu records=%zu hits=%u misses=%u mismatches=0\n", fixture.packed.size(),
                records.size(), hits, misses);
    return true;
}

bool check_gpu_alias_lookup_writes(ID3D12Device* device)
{
    AliasFixture fixture;
    if (!fixture.open(device)) return false;
    // Measured first and always printed. Wrong counts fail the test at the end, after the whole
    // fixture has run, so the output separates the add and sub words from every other check.
    const bool atomics_exact = array_atomics_exact(device, {ProbeArrayAtomics, sizeof(ProbeArrayAtomics)},
                                                   "array_atomics");
    enum : Word { Store = 0, Add = 1, Sub = 2, MinFloat = 3, MaxFloat = 4 };
    struct Record {
        std::uint64_t address;
        Word operation;
        Word operand;
    };
    static_assert(sizeof(Record) == 16);
    std::vector<Record> records;

    // Stores: words at the start, middle and end of every allocation (the placed allocation's
    // middle is the 4 GiB line itself) and the words on either side of the line. Every address is
    // distinct, so no two stores race; neighbouring words keep their pattern.
    std::vector<std::uint64_t> stored;
    for (const auto& title : fixture.titles)
        for (std::uint64_t address : {title->begin(), title->begin() + title->size / 2, title->last_word()})
            stored.push_back(address);
    stored.push_back(fixture.line() - 4);
    stored.push_back(fixture.line() + 4);
    for (std::size_t i = 0; i < stored.size(); ++i)
        for (std::size_t j = i + 1; j < stored.size(); ++j)
            if (stored[i] == stored[j]) throw std::runtime_error("two store records share an address");
    for (std::size_t i = 0; i < stored.size(); ++i)
        records.push_back({stored[i], Store, 0x5100'0000u + static_cast<Word>(i)});

    // Atomics: 256 threads each update one shared word per operation. The add and sub words sit
    // in the second and third allocations; the float words straddle the 4 GiB line's neighbours.
    constexpr Word contenders = 256;
    const std::uint64_t add_word = fixture.titles[1]->begin() + 0x100;
    const std::uint64_t sub_word = fixture.titles[2]->begin() + 0x200;
    const std::uint64_t min_word = fixture.line() + 0x40;
    const std::uint64_t max_word = fixture.line() - 0x40;
    word_at(add_word) = 0;
    word_at(sub_word) = 1'000'000;
    word_at(min_word) = float_bits(std::numeric_limits<float>::infinity());
    word_at(max_word) = float_bits(-std::numeric_limits<float>::infinity());
    Word sum = 0;
    float lowest = std::numeric_limits<float>::infinity();
    float highest = -std::numeric_limits<float>::infinity();
    for (Word i = 0; i < contenders; ++i) {
        const Word value = (i * 7919u) % 1000u + 1;
        const float number = static_cast<float>(static_cast<int>((i * 2654435761u) % 20001u) - 10000) / 8.0f;
        records.push_back({add_word, Add, value});
        records.push_back({sub_word, Sub, value});
        records.push_back({min_word, MinFloat, float_bits(number)});
        records.push_back({max_word, MaxFloat, float_bits(number)});
        sum += value;
        lowest = (std::min)(lowest, number);
        highest = (std::max)(highest, number);
    }
    // Rejected records must report 0 and change nothing, including bytes of a rejected word
    // that lie inside a range: stores to unmapped addresses, a store and a float min on the
    // placed range's tail-crossing word, and misaligned store, add and sub inside ranges.
    const std::uint64_t ghost = fixture.line() + 4 + (1ull << 32);
    const std::uint64_t tail = fixture.titles.back()->last_word() + 4;
    records.push_back({0x10, Store, miss});
    records.push_back({0xfffffffffffffff0ull, Store, miss});
    records.push_back({ghost, Store, miss});
    records.push_back({tail, Store, miss});
    records.push_back({tail, MinFloat, float_bits(-1.0e30f)});
    records.push_back({fixture.titles[0]->begin() + 2, Store, miss});
    records.push_back({add_word + 2, Add, 7});
    records.push_back({sub_word + 1, Sub, 7});
    records.push_back({fixture.titles[2]->begin() + fixture.titles[2]->range_bytes() - 2, Store, miss});
    constexpr Word rejected = 9;

    // Every word in every allocation, as the CPU expects it after the dispatch.
    std::vector<std::vector<Word>> expected(fixture.titles.size());
    for (std::size_t a = 0; a < fixture.titles.size(); ++a)
        expected[a].assign(static_cast<const Word*>(fixture.titles[a]->base),
                           static_cast<const Word*>(fixture.titles[a]->base) + fixture.titles[a]->size / 4);
    const auto expect_word = [&](std::uint64_t address, Word value) {
        for (std::size_t a = 0; a < fixture.titles.size(); ++a)
            if (address >= fixture.titles[a]->begin() && address < fixture.titles[a]->begin() + fixture.titles[a]->size)
                expected[a][(address - fixture.titles[a]->begin()) / 4] = value;
    };
    for (std::size_t i = 0; i < stored.size(); ++i) expect_word(stored[i], 0x5100'0000u + static_cast<Word>(i));
    expect_word(add_word, sum);
    expect_word(sub_word, 1'000'000 - sum);
    expect_word(min_word, float_bits(lowest));
    expect_word(max_word, float_bits(highest));
    std::vector<Word> expected_results(records.size());
    Word misses = 0;
    for (std::size_t r = 0; r < records.size(); ++r) {
        expected_results[r] = fixture.slot_of(records[r].address) ? 1 : 0;
        misses += expected_results[r] ? 0 : 1;
    }
    if (misses != rejected) throw std::runtime_error("fixture needs exactly the rejected records to miss");

    Kernel kernel(device, {WriteTitleAddresses, sizeof(WriteTitleAddresses)});
    const auto results = fixture.run(device, kernel, records, records.size());
    // Two groups, each with its own count and first mismatch: the shared add and sub words, which
    // depend on InterlockedAdd through the UAV array, and everything else (the per-record results,
    // then every other word: stores, float min and max, rejected records and untouched words).
    unsigned add_sub_mismatches = 0, other_mismatches = 0;
    const auto differs = [](unsigned& count, const char* what, Word actual, Word expected, unsigned record, unsigned field) {
        if (actual == expected) return;
        if (count++ == 0)
            std::fprintf(stderr, "%s record=%u field=%u expected=%08x actual=%08x\n", what, record, field, expected, actual);
    };
    for (std::size_t r = 0; r < records.size(); ++r)
        differs(other_mismatches, "result", results[r], expected_results[r], static_cast<unsigned>(r), 0);
    // Read back through the title pointers: no copy of the aliased memory was made.
    std::size_t words = 0;
    for (std::size_t a = 0; a < fixture.titles.size(); ++a) {
        const auto* actual = static_cast<const Word*>(fixture.titles[a]->base);
        for (std::size_t w = 0; w < expected[a].size(); ++w, ++words) {
            const auto* address = actual + w;
            if (address == &word_at(add_word))
                differs(add_sub_mismatches, "add_word", actual[w], expected[a][w], static_cast<unsigned>(a), static_cast<unsigned>(w));
            else if (address == &word_at(sub_word))
                differs(add_sub_mismatches, "sub_word", actual[w], expected[a][w], static_cast<unsigned>(a), static_cast<unsigned>(w));
            else
                differs(other_mismatches, "word", actual[w], expected[a][w], static_cast<unsigned>(a), static_cast<unsigned>(w));
        }
    }
    std::printf("gpu_alias_lookup_writes add_sub words=2 add=%08x sub=%08x expected_add=%08x expected_sub=%08x mismatches=%u\n",
                word_at(add_word), word_at(sub_word), sum, 1'000'000 - sum, add_sub_mismatches);
    std::printf("gpu_alias_lookup_writes other results=%zu words=%zu mismatches=%u\n", records.size(), words - 2,
                other_mismatches);
    std::printf("gpu_alias_lookup_writes ranges=%zu records=%zu stores=%zu contenders=%u sum=%u min=%g max=%g "
                "misses=%u words_checked=%zu mismatches=%u\n",
                fixture.packed.size(), records.size(), stored.size(), contenders, sum, lowest, highest, misses, words,
                add_sub_mismatches + other_mismatches);
    if (!atomics_exact)
        throw std::runtime_error("InterlockedAdd through a per-thread index into a UAV array is not applied exactly once "
                                 "per thread on this adapter (array_atomics counts above). This is an adapter defect, "
                                 "recorded for the Windows 10 in-box WARP 10.0.19041 in docs/experiments.md");
    if (add_sub_mismatches + other_mismatches != 0)
        throw std::runtime_error("gpu_alias_lookup_writes differs from the CPU reference");
    return true;
}
