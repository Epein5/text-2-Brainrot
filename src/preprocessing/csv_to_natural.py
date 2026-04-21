def create_finetuning_prompt(normal_text: str, brain_text: str) -> str:
    """Convert normal text to brainrot text for instructional finetuning."""
    return f"Instruction: Convert the following text to brainrot slang.\nInput: {normal_text}\nOutput: {brain_text}"