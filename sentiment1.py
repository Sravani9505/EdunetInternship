import os
from groq import Groq

# Set API Key directly (NOT recommended for production)
API_KEY = "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"

# Alternatively, set it as an environment variable
os.environ["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"] = API_KEY

if not API_KEY:
    raise ValueError("Missing Groq API key. Set GROQ_API_KEY in your environment variables.")

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Sample dataset: (Text, Actual Sentiment)
dataset = [
    ("I love this product!", "positive"),  # Clear positive
    ("This is the worst experience ever.", "negative"),  # Clear negative
    ("It's okay, nothing special.", "neutral"),  # Neutral
    ("Amazing quality and great customer service!", "positive"),  # Positive
    ("I'm not happy with this purchase.", "negative"),  # Negative
    ("I thought it would be great, but I'm disappointed.", "negative"),  # Negative
    ("Not bad, but I expected more.", "neutral"),  # Neutral
    ("The packaging was good, but the product was terrible.", "negative"),  # Negative
    ("Wow! This exceeded all my expectations!", "positive"),  # Positive
    ("Meh, I’ve seen better.", "neutral"),  # Neutral
    ("This is exactly what I wanted!", "positive"),  # Positive
    ("The service was slow, but the food was great.", "neutral"),  # Mixed
    ("I can't believe how awful this is.", "negative"),  # Negative
    ("I was hoping for better, but it's just okay.", "neutral"),  # Neutral
    ("Absolutely fantastic! Highly recommended!", "positive"),  # Positive
    ("Terrible quality. Would not buy again.", "negative"),  # Negative
    ("It’s fine, nothing to complain about.", "neutral"),  # Neutral
    ("Oh great, another delay!", "negative"),  # Sarcasm (should be negative)
    ("I don’t hate it, but I wouldn’t buy it again.", "negative"),  # Negation-based negative
    ("The movie was so bad, it was actually funny.", "neutral"),  # Mixed sentiment
    ("The app crashes sometimes, but I love its features!", "neutral"),  # Mixed sentiment
    ("I’m over the moon! Couldn’t be happier!", "positive"),  # Strongly positive
    ("You call this customer service? What a joke.", "negative"),  # Sarcasm, negative
    ("Well, I didn’t love it, but I’ve seen worse.", "neutral"),  # Neutral
    ("It’s the best worst thing I’ve ever bought.", "neutral"),  # Confusing phrasing
]


correct_predictions = 0

for text, actual_sentiment in dataset:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a sentiment analyst. Reply with 'positive', 'negative', or 'neutral' only."},
                {"role": "user", "content": f"Analyze the sentiment: '{text}'"}
            ],
            model="llama3-8b-8192"
        )

        # Correct way to access response content
        predicted_sentiment = chat_completion.choices[0].message.content.strip().lower()

        # Compare with actual sentiment
        if predicted_sentiment == actual_sentiment:
            correct_predictions += 1

        print(f"Text: {text}\nPredicted: {predicted_sentiment}, Actual: {actual_sentiment}\n")

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")

# Calculate accuracy
accuracy = (correct_predictions / len(dataset)) * 100
print(f"Sentiment Analysis Accuracy: {accuracy:.2f}%")
