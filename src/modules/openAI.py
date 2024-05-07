import openai

class OpenAI:
    def __init__(self, api_key):
        """
        Initializes the TextClassifier with the provided OpenAI API key.
        """
        openai.api_key = api_key
    
    def classify_openai(self, text, model="gpt-3.5-turbo"):
        """
        Classifies the given text using the specified model.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bitte ordnen Sie den folgenden Text einer dieser Kategorien zu: Interview, Leserbrief, Kommentar oder keine von drei. Bei Interview geht es haupsächlich um die Beantwortung von Fragen, often über sich selbst, manchmal auch in Form eines Dialogs. Ein Leserbrief ist eine Meinungsäußerung auf den Inhalt eines Artikel, oft länger als Kommentar. Kommentar ist oft ein kurzen Meinungsbeitrag. Zu keiner der drei Kategorien können einfache info Nachrichte gehören. Als Antwort geben Sie nur Wert '0', wenn das ein Interview ist, '2' wenn ein Leserbrief, '3' wenn ein Kommentar oder '1' wenn keine von diesen Kategorien an."},
                {"role": "user", "content": text}
            ]
        )
        category = response.choices[0].message['content'].strip()
        return category

    def classify_openai_gpt4(self, text):
        """
        Classifies the given text using the GPT-4 model.
        """
        return self.classify_openai(text, model="gpt-4-turbo-preview")
