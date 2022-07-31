# Chatbot
This Chatbot support English, French, Arabic and soon tunisian dialect. 

I strongly recommend working in an isolated environment so you can work on different projects without having conflicting library versions (because you may need a different version of frameworks or languages that you already have!)

For every single message from the user, the chatbot has to figure out the language and respond to this message based on the "intents" file for it. (Please feel free to add more intents)

```mermaid
flowchart LR
   Input == LanguageDetection ==> English ==  EnglishModel ==> FinalResponse
   Input == LanguageDetection ==> French ==  FrenchModel ==> FinalResponse
   Input == LanguageDetection ==> Arabic ==  ArabicModel ==> FinalResponse
   
```

Once you clone the project you can try it out:
 ```
 git clone https://github.com/MosbehBarhoumi/Chatbot.git
 cd Chatbot 
 python app.py
```
![english](https://user-images.githubusercontent.com/78423450/182024762-a32f95cd-8ad6-4e06-8322-3cc3ed6c0e05.png)
![french](https://user-images.githubusercontent.com/78423450/182024763-da0208f0-c306-4383-905c-a75426d7db45.png)
![Arabic](https://user-images.githubusercontent.com/78423450/182024765-3b9aee94-432b-4af4-a704-012805cd150d.png)


