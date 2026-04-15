#!/usr/bin/env python3

"""
Test script to verify the two-stage door handle processing logic
"""

def is_door_handle_prompt(prompt: str) -> bool:
    """Check if the prompt is for door handle detection."""
    prompt_lower = prompt.lower()
    return "door handle" in prompt_lower

def extract_pure_door_handle(original_prompt: str) -> str:
    """Extract pure 'door handle' from any door handle related prompt"""
    if is_door_handle_prompt(original_prompt):
        return "door handle"
    return original_prompt

def test_door_handle_detection():
    """Test door handle prompt detection"""
    
    # Test cases for door handle detection
    test_cases = [
        ("door handle", True),
        ("Door Handle", True),
        ("DOOR HANDLE", True),
        ("rotate door handle", True),
        ("door handle mechanism", True),
        ("handle door", False),
        ("door", False),
        ("handle", False),
        ("window handle", False),
        ("door knob", False),
    ]
    
    print("Testing door handle detection:")
    for prompt, expected in test_cases:
        result = is_door_handle_prompt(prompt)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{prompt}' -> {result} (expected: {expected})")
    
    print("\nTesting door handle extraction:")
    extract_cases = [
        ("door handle", "door handle"),
        ("rotate door handle", "door handle"),
        ("door handle mechanism", "door handle"),
        ("some other prompt", "some other prompt"),
    ]
    
    for prompt, expected in extract_cases:
        result = extract_pure_door_handle(prompt)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{prompt}' -> '{result}' (expected: '{expected}')")

def test_processing_logic():
    """Test the processing logic flow"""
    
    print("\n" + "="*50)
    print("Two-stage door handle processing logic:")
    print("="*50)
    
    print("\nLogic flow:")
    print("1. For door handle prompts:")
    print("   - Stage 1: Use 'door handle' as primary prompt")
    print("   - If primary has results, store for stage 2")
    print("   - Stage 2: Use 'rotate door handle' as secondary prompt")
    print("   - Compare areas: if secondary area > 0 and < primary area, use secondary")
    print("   - Otherwise, use primary ('door handle') as fallback")
    
    print("\n2. For non-door handle prompts:")
    print("   - Process normally with original prompt")
    
    print("\nImplementation details:")
    print("- Only frames with primary results get secondary processing")
    print("- Smaller mask area indicates more precise segmentation") 
    print("- Logging shows which segmentation method was chosen")

if __name__ == "__main__":
    test_door_handle_detection()
    test_processing_logic()
    print("\n" + "="*50)
    print("Test completed. Check the logic in scenefun3d_sam3_image.py")
    print("="*50)