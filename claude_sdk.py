import anyio
from claude_code.sdk import query
async def main():
    prompt = "What is the capital of France?"
    async for message in query(prompt):
        print(message)
    anyio.run(main)