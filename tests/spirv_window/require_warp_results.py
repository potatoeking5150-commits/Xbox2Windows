"""Fail unless every SPIR-V bridge WARP test ran and passed in a CTest JUnit report.

Usage: require_warp_results.py <ctest-junit.xml>

Hosted CI runs this after CTest so that a green run proves each required test ran. A test that is
missing from the report, skipped or failed fails this check.
"""
import sys
import xml.etree.ElementTree as ET

REQUIRED = [f"spirv_{name}_warp" for name in ("address_window", "sampler_views", "ballot_extract",
                                               "compaction", "compare_sampler", "minlod", "minlod_levels")]


def statuses(junit):
    result = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        state = {"run": "passed", "fail": "failed", "notrun": "skipped"}.get(case.get("status"), "failed")
        if case.find("failure") is not None or case.find("error") is not None:
            state = "failed"
        elif case.find("skipped") is not None:
            state = "skipped"
        result[case.get("name")] = state
    return result


def main(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    found = statuses(argv[0])
    problems = [f"{name}: {found.get(name, 'not in the report')}" for name in REQUIRED if found.get(name) != "passed"]
    for name in REQUIRED:
        print(f"{name}: {found.get(name, 'not in the report')}")
    if problems:
        print("error: SPIR-V bridge WARP tests did not all run and pass", file=sys.stderr)
        return 1
    print(f"all {len(REQUIRED)} SPIR-V bridge WARP tests ran and passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
