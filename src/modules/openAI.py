import openai

class OpenAI:
    def __init__(self, api_key):
        """
        Initializes the TextClassifier with the provided OpenAI API key.
        """
        openai.api_key = api_key
    
    def classify_openai(self, text, model="gpt-3.5-turbo-0125"):
        """
        Classifies the given text using the specified model.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You will be given a German text sample. Classify it into one of four categories: Interview, Letter to Editor, Opinion, or Other. Output only the label: 1 for Interview, 2 for Letter to Editor, 3 for Opinion, and 0 for Other. Use the following classification criteria: 1. Interview: Structure: dialog format, question-answer pairs, uses names and titles to indicate speakers, can vary in length but typically provides substantial responses. Language: lot of question marks, first- and third-person pronouns; questions are often followed by colons or quotation marks; uses names and titles to indicate speakers . Content: intro of interviewee, topic-focused Q&A: interviewee’s expertise, experiences, or opinions. 2. Letter to Editor: Structure: Heading, starts with a salutation or direct address; often labeled 'Leserbrief' or 'Leserpost'; generally short and concise, one or few paragraphs. Language: formal, persuasive; exclamation marks, rhetorical questions; strongly opinionated language. Content: personal opinions or reactions to specific articles/events/societal issues. 3. Opinion: Structure: essay-like with intro, body, conclusion; often begins with a clear statement or provocative question; multiple paragraphs, may include subheadings. Language: persuasive, formal, emotive; first-person pronouns. Content: analysis of current events, societal issues, political matters, or cultural topics; and arguments supported by evidence, examples, references. 0. Other: Structure: varied vormats, including news, events, announcements, and practical advice; diverse formatting with headings, bullet points, lists. Language: neutral or factual language for news; persuasive for advertisements. Content: wide-range, event-driven. Instruction: Given a text sample, classify it based on the criteria above. Output only the corresponding label: 1 for Interview, 2 for Letter to Editor, 3 for Opinion, and 0 for Other."},
                {"role": "user", "content": text}
            ]
        )
        category = response.choices[0].message['content'].strip()
        return category

    def classify_openai_gpt4o(self, text):
        """
        Classifies the given text using the GPT-4 model.
        """
        return self.classify_openai(text, model="gpt-4o")
