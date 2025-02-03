import os
from groq import Groq

# Use your API key directly or set it as an environment variable
API_KEY = os.environ.get("GROQ_API_KEY", "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")

client = Groq(api_key=API_KEY)

# Ask the user for input text
user_text = input("Enter text for sentiment analysis: ")

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a sentiment analyst."},
        {"role": "user", "content": f"Analyze the sentiment: '{user_text}'"}
    ],
    model="llama3-8b-8192",
)

# Print the sentiment analysis result
print("Sentiment Analysis Result:", chat_completion.choices[0].message.content)
