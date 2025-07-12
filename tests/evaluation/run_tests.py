#!/usr/bin/env python3
"""
Test runner for evaluation module tests.
Runs all tests and provides a comprehensive report.
"""

import sys
import traceback
from pathlib import Path

# Add the project root to the path so we can import the test modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tests.evaluation.test_cli_consistency import (
        TestArgumentValidation,
        TestCLIConsistency,
        TestIntegrationScenarios,
    )
    from tests.evaluation.test_path_handling import (
        TestBasicPathHandling,
        TestConfigIntegration,
        TestEdgeCasesAndErrors,
        TestPathResolution,
        TestPlaceholderExpansion,
    )
except ImportError as e:
    print(f"❌ Failed to import test modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def run_test_class(test_class):
    """Run all test methods in a test class."""
    instance = test_class()
    methods = [method for method in dir(instance) if method.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in methods:
        try:
            print(f"    Running {method_name}...", end=" ")
            getattr(instance, method_name)()
            print("✅")
            passed += 1
        except Exception as e:
            print("❌")
            print(f"      Error: {e}")
            if "--verbose" in sys.argv:
                print(f"      Traceback: {traceback.format_exc()}")
            failed += 1

    return passed, failed


def main():
    """Run all tests and provide a comprehensive report."""
    print("🧪 Running Evaluation Module Test Suite\n")

    test_classes = [
        # Basic functionality tests
        ("Basic Path Handling", TestBasicPathHandling),
        ("Placeholder Expansion", TestPlaceholderExpansion),
        ("Path Resolution", TestPathResolution),
        ("Edge Cases & Errors", TestEdgeCasesAndErrors),
        ("Config Integration", TestConfigIntegration),
        # CLI and integration tests
        ("CLI Consistency", TestCLIConsistency),
        ("Argument Validation", TestArgumentValidation),
        ("Integration Scenarios", TestIntegrationScenarios),
    ]

    total_passed = 0
    total_failed = 0

    for test_name, test_class in test_classes:
        print(f"📋 {test_name}:")
        try:
            passed, failed = run_test_class(test_class)
            total_passed += passed
            total_failed += failed

            if failed == 0:
                print(f"  ✅ All {passed} tests passed\n")
            else:
                print(f"  ⚠️  {passed} passed, {failed} failed\n")

        except Exception as e:
            print(f"  ❌ Failed to run test class: {e}\n")
            total_failed += 1

    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"  Total tests run: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("  🎉 All tests passed! Path handling is working correctly.")
        return 0
    else:
        print(f"  ⚠️  {total_failed} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python run_tests.py [--verbose]")
        print("  --verbose  Show full tracebacks for failed tests")
        sys.exit(0)

    exit_code = main()
    sys.exit(exit_code)
