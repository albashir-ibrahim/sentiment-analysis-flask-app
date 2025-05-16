import requests
import json

def sentiment_analyzer(text_to_analyse):
    """
    Analyzes the sentiment of the input text using Watson NLP service.
    """
    # URL for the Watson NLP sentiment analysis service
    url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'
    
    # Headers required for the API request
    # As per document and refined by for the model ID format
    headers = {
        "grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"
    }
    
    # Input JSON payload for the API
    # As per document 
    input_json = {
        "raw_document": {
            "text": text_to_analyse
        }
    }
    
    # Make the POST request to the API
    # As per document 
    response = requests.post(url, json=input_json, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response text (which is a JSON string) into a Python dictionary
        # As per document 
        try:
            formatted_response = json.loads(response.text)
            
            # Extract the label and score from the response
            # As per document 
            label = formatted_response['documentSentiment']['label']
            score = formatted_response['documentSentiment']['score']
            
            return {'label': label, 'score': score}
        except (KeyError, json.JSONDecodeError) as e:
            # Handle cases where the response format is not as expected or JSON is malformed
            print(f"Error processing response: {e}")
            return {'label': None, 'score': None} # Modified for error handling (anticipating Task 7)
            
    elif response.status_code == 500: # As per document for invalid entries
        print(f"Received 500 Internal Server Error: {response.text}")
        return {'label': None, 'score': None} # Modified for error handling (anticipating Task 7)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        return {'label': None, 'score': None}

# Quick test (optional, can be removed later)
if __name__ == '__main__':
    test_text_positive = "I love this new technology!"
    result_positive = sentiment_analyzer(test_text_positive)
    print(f"Sentiment for '{test_text_positive}': {result_positive}")

    test_text_negative = "I hate dealing with this problem."
    result_negative = sentiment_analyzer(test_text_negative)
    print(f"Sentiment for '{test_text_negative}': {result_negative}")

    test_text_neutral = "The sky is blue today." # This model might not have a strong neutral category, let's see
    result_neutral = sentiment_analyzer(test_text_neutral)
    print(f"Sentiment for '{test_text_neutral}': {result_neutral}")
    
    test_text_invalid = "" # Testing blank input early (anticipating Task 7/Optional Exercise 3)
    result_invalid = sentiment_analyzer(test_text_invalid)
    print(f"Sentiment for BLANK TEXT: {result_invalid}")