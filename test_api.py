import os
import asyncio
import aiohttp

async def test_api():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API key found")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say 'API test successful'"}],
        "max_tokens": 50
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"Success: {data['choices'][0]['message']['content']}")
            else:
                error = await resp.text()
                print(f"Error {resp.status}: {error}")

if __name__ == "__main__":
    asyncio.run(test_api())
