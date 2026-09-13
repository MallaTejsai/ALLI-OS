#!/usr/bin/env python3
"""
Comprehensive real-user end-to-end tests for ALLI v0.6.

Tests the complete memory lifecycle: write, read, update, conflict,
restart persistence, semantic retrieval, irrelevant filtering.
"""
import tempfile
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.foundation.memory_engine import MemoryEngine


def new_engine(allow_qwen=True):
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test.sqlite")
    eng = MemoryEngine(db_path=db_path, allow_qwen=allow_qwen)
    return eng, test_dir


def test_basic_memory():
    """Test: My name is X → What is my name?"""
    eng, td = new_engine(allow_qwen=False)
    try:
        eng.remember("My name is Tejsai.")
        ans = eng.recall("What is my name?")
        print(f"  [BASIC] {ans}")
        assert "tejsai" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_semantic_retrieval():
    """Test: Career goal stored, asked with different wording"""
    eng, td = new_engine()
    try:
        eng.remember("My long-term career goal is to become an AI engineer.")
        # Different wording
        ans = eng.recall("What kind of career am I aiming for?")
        print(f"  [SEMANTIC] {ans}")
        assert "ai engineer" in ans.lower() or "artificial intelligence" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_preference():
    """Test: Preference stored and retrieved"""
    eng, td = new_engine()
    try:
        eng.remember("I prefer learning through practical projects.")
        ans = eng.recall("How do I prefer to learn?")
        print(f"  [PREF] {ans}")
        assert "practical" in ans.lower() or "project" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_goal():
    """Test: Goal stored and retrieved with different wording"""
    eng, td = new_engine()
    try:
        eng.remember("I want to become an AI engineer.")
        ans = eng.recall("What is my professional goal?")
        print(f"  [GOAL] {ans}")
        assert "ai engineer" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_project():
    """Test: Project stored and retrieved indirectly"""
    eng, td = new_engine()
    try:
        eng.remember("I am building a personal AI assistant called ALLI.")
        ans = eng.recall("What project am I working on?")
        print(f"  [PROJECT] {ans}")
        assert "alli" in ans.lower() or "ai assistant" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_explicit_remember():
    """Test: Explicit remember request survives restart"""
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "persist.sqlite")
    try:
        eng = MemoryEngine(db_path=db_path, allow_qwen=False)
        eng.remember("Remember that I want to learn Java.")
        eng.close()

        # Restart
        eng2 = MemoryEngine(db_path=db_path, allow_qwen=False)
        ans = eng2.recall("What language do I want to learn?")
        print(f"  [EXPLICIT RESTART] {ans}")
        assert "java" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng2.close()
        shutil.rmtree(td)


def test_temporary_info():
    """Test: Temporary info does NOT become permanent"""
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "temp.sqlite")
    try:
        eng = MemoryEngine(db_path=db_path, allow_qwen=False)
        eng.remember("I am debugging this function right now.")
        eng.close()

        # Restart - should NOT remember the temporary statement
        eng2 = MemoryEngine(db_path=db_path, allow_qwen=False)
        ans = eng2.recall("What was I debugging?")
        print(f"  [TEMPORARY] {ans}")
        eng2.close()
        # Should say unknown, not recall the temporary info
        assert "don't" in ans.lower() or "unknown" in ans.lower() or "no" in ans.lower(), f"FAIL: {ans}"
        assert "debugging" not in ans.lower(), f"FAIL - remembered temporary info: {ans}"
        print("  ✓ PASS")
    finally:
        shutil.rmtree(td)


def test_update():
    """Test: Update - Python → Java should reflect current state"""
    eng, td = new_engine()
    try:
        eng.remember("I am learning Python.")
        eng.remember("I finished learning Python and now I am learning Java.")
        ans = eng.recall("What programming language am I currently learning?")
        print(f"  [UPDATE] {ans}")
        # Should favor Java as current state
        assert "java" in ans.lower(), f"FAIL: {ans}"
        # Should NOT say both without understanding the update
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_conflict():
    """Test: Conflict - want Java → don't want Java anymore"""
    eng, td = new_engine()
    try:
        eng.remember("I want to learn Java.")
        eng.remember("I don't want to learn Java anymore.")
        ans = eng.recall("Do I want to learn Java?")
        print(f"  [CONFLICT] {ans}")
        # Should correctly resolve to current negative preference
        assert "don't" in ans.lower() or "not" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_restart_persistence():
    """Test: Full restart persistence"""
    td = tempfile.mkdtemp()
    db_path = os.path.join(td, "restart.sqlite")
    try:
        eng = MemoryEngine(db_path=db_path, allow_qwen=False)
        eng.remember("My favorite color is blue.")
        eng.remember("I am learning Python.")
        eng.remember("I want to become an AI engineer.")
        eng.close()

        # Restart
        eng2 = MemoryEngine(db_path=db_path, allow_qwen=False)
        ans1 = eng2.recall("What is my favorite color?")
        ans2 = eng2.recall("What am I learning?")
        ans3 = eng2.recall("What is my goal?")
        print(f"  [RESTART 1] {ans1}")
        print(f"  [RESTART 2] {ans2}")
        print(f"  [RESTART 3] {ans3}")
        assert "blue" in ans1.lower()
        assert "python" in ans2.lower()
        assert "ai engineer" in ans3.lower()
        print("  ✓ PASS")
    finally:
        eng2.close()
        shutil.rmtree(td)


def test_different_wording():
    """Test: Semantic retrieval with different wording"""
    eng, td = new_engine()
    try:
        eng.remember("I am learning Flutter for mobile development.")
        ans = eng.recall("What mobile development technology am I interested in?")
        print(f"  [DIFF WORDING] {ans}")
        assert "flutter" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_irrelevant_memory():
    """Test: Unrelated memories don't pollute context"""
    eng, td = new_engine()
    try:
        # Create unrelated memories
        eng.remember("I like pizza.")
        eng.remember("I have a Samsung phone.")
        eng.remember("My goal is to become an AI engineer.")
        eng.remember("I prefer learning through projects.")

        # Ask about only one topic
        ans = eng.recall("What is my goal?")
        print(f"  [IRRELEVANT] {ans}")
        # Should NOT mention pizza, phone, learning preference
        assert "ai engineer" in ans.lower(), f"FAIL: {ans}"
        assert "pizza" not in ans.lower()
        assert "samsung" not in ans.lower()
        assert "project" not in ans.lower()
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_multiple_memories():
    """Test: Multiple memories around different subjects"""
    eng, td = new_engine()
    try:
        eng.remember("I like Python.")
        eng.remember("I am learning Java.")
        eng.remember("My goal is to become an AI engineer.")
        eng.remember("I am building ALLI.")
        eng.remember("My favorite color is blue.")

        ans1 = eng.recall("What do I like?")
        ans2 = eng.recall("What am I learning?")
        ans3 = eng.recall("What is my goal?")
        ans4 = eng.recall("What am I building?")
        ans5 = eng.recall("What is my favorite color?")

        print(f"  [MULTI 1] {ans1}")
        print(f"  [MULTI 2] {ans2}")
        print(f"  [MULTI 3] {ans3}")
        print(f"  [MULTI 4] {ans4}")
        print(f"  [MULTI 5] {ans5}")

        assert "python" in ans1.lower()
        assert "java" in ans2.lower()
        assert "ai engineer" in ans3.lower()
        assert "alli" in ans4.lower()
        assert "blue" in ans5.lower()
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_unknown_information():
    """Test: Unknown info handled honestly (no hallucination)"""
    eng, td = new_engine()
    try:
        ans = eng.recall("What was my favorite movie?")
        print(f"  [UNKNOWN] {ans}")
        # Should NOT invent a favorite movie
        assert "don't" in ans.lower() or "unknown" in ans.lower() or "no" in ans.lower() or "stored" in ans.lower(), f"FAIL: {ans}"
        assert "movie" in ans.lower() or "information" in ans.lower(), f"FAIL: {ans}"
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_large_memory_set():
    """Test: Large memory set - retrieval remains useful"""
    eng, td = new_engine()
    try:
        # Add many memories
        for i in range(50):
            eng.remember(f"I like technology {i}.")
        for i in range(20):
            eng.remember(f"I am learning framework {i}.")
        eng.remember("My main goal is to master AI engineering.")
        eng.remember("I am building a project called ALLI-OS.")

        ans = eng.recall("What is my main goal?")
        print(f"  [LARGE SET] {ans}")
        assert "ai engineering" in ans.lower() or "master ai" in ans.lower(), f"FAIL: {ans}"
        # Irrelevant memories should not leak
        assert "technology 5" not in ans.lower()
        assert "framework 10" not in ans.lower()
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_entity_isolation():
    """Test: Entity isolation - mom vs me"""
    eng, td = new_engine()
    try:
        eng.remember("My mom likes pink.")
        eng.remember("I like blue.")

        ans1 = eng.recall("What color does my mom like?")
        ans2 = eng.recall("What color do I like?")
        ans3 = eng.recall("Who likes pink?")

        print(f"  [ENTITY 1] {ans1}")
        print(f"  [ENTITY 2] {ans2}")
        print(f"  [ENTITY 3] {ans3}")

        assert "pink" in ans1.lower()
        assert "blue" in ans2.lower()
        # Reverse query - should mention mom relationship
        assert "mom" in ans3.lower() or "mother" in ans3.lower()
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def test_general_knowledge_routing():
    """Test: General knowledge questions don't use personal memory"""
    eng, td = new_engine()
    try:
        eng.remember("I like Python.")
        ans = eng.recall("What is Python?")
        print(f"  [GENERAL] {ans[:100]}...")
        # Should give general explanation, NOT personal preference
        assert "programming language" in ans.lower() or "language" in ans.lower(), f"FAIL: {ans}"
        # Should NOT say "You like Python"
        assert "you like" not in ans.lower()
        print("  ✓ PASS")
    finally:
        eng.close()
        shutil.rmtree(td)


def run_all_tests():
    tests = [
        ("Basic Memory", test_basic_memory),
        ("Semantic Retrieval", test_semantic_retrieval),
        ("Preference", test_preference),
        ("Goal", test_goal),
        ("Project", test_project),
        ("Explicit Remember + Restart", test_explicit_remember),
        ("Temporary Info (no persist)", test_temporary_info),
        ("Update (Python -> Java)", test_update),
        ("Conflict Resolution", test_conflict),
        ("Restart Persistence", test_restart_persistence),
        ("Different Wording", test_different_wording),
        ("Irrelevant Memory Filtering", test_irrelevant_memory),
        ("Multiple Memories", test_multiple_memories),
        ("Unknown Information", test_unknown_information),
        ("Large Memory Set", test_large_memory_set),
        ("Entity Isolation", test_entity_isolation),
        ("General Knowledge Routing", test_general_knowledge_routing),
    ]

    results = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
            results.append((name, True, None))
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            results.append((name, False, str(e)))

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}" + (f" - {err}" if err else ""))
    print(f"\nTotal: {passed}/{len(results)} passed")
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)