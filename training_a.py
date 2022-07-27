# -------------- Importing the libraries ------------------- 

import nltk
# CAMeL Tools is suite of Arabic natural language processing tools
# developed by the CAMeL Lab at New York University Abu Dhabi.
from nltk.stem.isri import ISRIStemmer #arabic stemmer
from nltk.stem import WordNetLemmatizer

from camel_tools.tokenizers.word import simple_word_tokenize #arabic tokenizer
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
import json
import pickle
import numpy as np
import random


# importing Keras libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation , Dropout
from tensorflow.keras.optimizers import SGD

# -------------- load the intents_a -------------------

data_file = open('intentsa.json').read()
intentsa = json.loads(data_file)

# -------------- Cleaning the text -------------------
#                preprocess data


lemmatizer = WordNetLemmatizer()

# create a list of all the words in the file 
words=[]
#create a list of classes for our tags
classes = []
# craate teh corpus after cleaning the text
corpus = []
ignore_chars = ['؟', '!','،',',','.','.','?']



# loop through the intents_a.json file
for data in intentsa['intentsa']:
    for pattern in data['patterns']:
    # remove stopwords with a for loop
    # after cleaining we'll add it to corpus    
    
        #tokenize each statement into words
        w = nltk.word_tokenize(pattern)
        # add the tokenize words to words[]
        words.extend(w)
        #add each tokenized words to its tag e.g., ['أهلا','مرحبا'] greeting
        corpus.append((w, data['tag']))
        
        # add all the tags to our classes list
        if data['tag'] not in classes:
            classes.append(data['tag'])

#return the root of the word        
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]


#------------------


#Arabic_stop_words = list(stopwords.words("arabic"))


#for w in words:
 #   if w in Arabic_stop_words:
  #      words.remove(w)
   # # Normalize alef variants to 'ا'
    #nw = normalize_alef_ar(w)
    # Normalize alef maksura 'ى' to yeh 'ي'
  #  nw = normalize_alef_maksura_ar(w)
   # # Normalize teh marbuta 'ة' to heh 'ه'
    #nw = normalize_teh_marbuta_ar(w)
    # removing Arabic diacritical marks
    #nw = dediac_ar(w)
    #words.extend(nw)

stemmer = ISRIStemmer() 

#words = [stemmer.stem(w) for w in words if w not in ignore_chars]

Arabic_stop_words= ['', '،', 'ء', 'ءَ', 'آ', 'آب', 'آذار', 'آض', 'آل', 'آمينَ', 'آناء', 'آنفا', 'آه', 'آهاً', 'آهٍ', 'آهِ', 'أ', 'أبدا', 'أبريل', 'أبو', 'أبٌ', 'أجل', 'أجمع', 'أحد', 'أخبر', 'أخذ', 'أخو', 'أخٌ', 'أربع', 'أربعاء', 'أربعة', 'أربعمئة', 'أربعمائة', 'أرى', 'أسكن', 'أصبح', 'أصلا', 'أضحى', 'أطعم', 'أعطى', 'أعلم', 'أغسطس', 'أفريل', 'أفعل به', 'أفٍّ', 'أقبل', 'أكتوبر', 'أل', 'ألا', 'ألف', 'ألفى', 'أم', 'أما', 'أمام', 'أمامك', 'أمامكَ', 'أمد', 'أمس', 'أمسى', 'أمّا', 'أن', 'أنا', 'أنبأ', 'أنت', 'أنتم', 'أنتما', 'أنتن', 'أنتِ', 'أنشأ', 'أنه', 'أنًّ', 'أنّى', 'أهلا', 'أو', 'أوت', 'أوشك', 'أول', 'أولئك', 'أولاء', 'أولالك', 'أوّهْ', 'أى', 'أي', 'أيا', 'أيار', 'أيضا', 'أيلول', 'أين', 'أيّ', 'أيّان', 'أُفٍّ', 'ؤ', 'إحدى', 'إذ', 'إذا', 'إذاً', 'إذما', 'إذن', 'إزاء', 'إلى', 'إلي', 'إليكم', 'إليكما', 'إليكنّ', 'إليكَ', 'إلَيْكَ', 'إلّا', 'إمّا', 'إن', 'إنَّ', 'إى', 'إياك', 'إياكم', 'إياكما', 'إياكن', 'إيانا', 'إياه', 'إياها', 'إياهم', 'إياهما', 'إياهن', 'إياي', 'إيهٍ', 'ئ', 'ا', 'ا?', 'ا?ى', 'االا', 'االتى', 'ابتدأ', 'ابين', 'اتخذ', 'اثر', 'اثنا', 'اثنان', 'اثني', 'اثنين', 'اجل', 'احد', 'اخرى', 'اخلولق', 'اذا', 'اربعة', 'اربعون', 'اربعين', 'ارتدّ', 'استحال', 'اصبح', 'اضحى', 'اطار', 'اعادة', 'اعلنت', 'اف', 'اكثر', 'اكد', 'الآن', 'الألاء', 'الألى', 'الا', 'الاخيرة', 'الان', 'الاول', 'الاولى', 'التى', 'التي', 'الثاني', 'الثانية', 'الحالي', 'الذاتي', 'الذى', 'الذي', 'الذين', 'السابق', 'الف', 'اللاتي', 'اللتان', 'اللتيا', 'اللتين', 'اللذان', 'اللذين', 'اللواتي', 'الماضي', 'المقبل', 'الوقت', 'الى', 'الي', 'اليه', 'اليها', 'اليوم', 'اما', 'امام', 'امس', 'امسى', 'ان', 'انبرى', 'انقلب', 'انه', 'انها', 'او', 'اول', 'اي', 'ايار', 'ايام', 'ايضا', 'ب', 'بؤسا', 'بإن', 'بئس', 'باء', 'بات', 'باسم', 'بان', 'بخٍ', 'بد', 'بدلا', 'برس', 'بسبب', 'بسّ', 'بشكل', 'بضع', 'بطآن', 'بعد', 'بعدا', 'بعض', 'بغتة', 'بل', 'بلى', 'بن', 'به', 'بها', 'بهذا', 'بيد', 'بين', 'بَسْ', 'بَلْهَ', 'ة', 'ت', 'تاء', 'تارة', 'تاسع', 'تانِ', 'تانِك', 'تبدّل', 'تجاه', 'تحت', 'تحوّل', 'تخذ', 'ترك', 'تسع', 'تسعة', 'تسعمئة', 'تسعمائة', 'تسعون', 'تسعين', 'تشرين', 'تعسا', 'تعلَّم', 'تفعلان', 'تفعلون', 'تفعلين', 'تكون', 'تلقاء', 'تلك', 'تم', 'تموز', 'تينك', 'تَيْنِ', 'تِه', 'تِي', 'ث', 'ثاء', 'ثالث', 'ثامن', 'ثان', 'ثاني', 'ثلاث', 'ثلاثاء', 'ثلاثة', 'ثلاثمئة', 'ثلاثمائة', 'ثلاثون', 'ثلاثين', 'ثم', 'ثمان', 'ثمانمئة', 'ثمانون', 'ثماني', 'ثمانية', 'ثمانين', 'ثمنمئة', 'ثمَّ', 'ثمّ', 'ثمّة', 'ج', 'جانفي', 'جدا', 'جعل', 'جلل', 'جمعة', 'جميع', 'جنيه', 'جوان', 'جويلية', 'جير', 'جيم', 'ح', 'حاء', 'حادي', 'حار', 'حاشا', 'حاليا', 'حاي', 'حبذا', 'حبيب', 'حتى', 'حجا', 'حدَث', 'حرى', 'حزيران', 'حسب', 'حقا', 'حمدا', 'حمو', 'حمٌ', 'حوالى', 'حول', 'حيث', 'حيثما', 'حين', 'حيَّ', 'حَذارِ', 'خ', 'خاء', 'خاصة', 'خال', 'خامس', 'خبَّر', 'خلا', 'خلافا', 'خلال', 'خلف', 'خمس', 'خمسة', 'خمسمئة', 'خمسمائة', 'خمسون', 'خمسين', 'خميس', 'د', 'دال', 'درهم', 'درى', 'دواليك', 'دولار', 'دون', 'دونك', 'ديسمبر', 'دينار', 'ذ', 'ذا', 'ذات', 'ذاك', 'ذال', 'ذانك', 'ذانِ', 'ذلك', 'ذهب', 'ذو', 'ذيت', 'ذينك', 'ذَيْنِ', 'ذِه', 'ذِي', 'ر', 'رأى', 'راء', 'رابع', 'راح', 'رجع', 'رزق', 'رويدك', 'ريال', 'ريث', 'رُبَّ', 'ز', 'زاي', 'زعم', 'زود', 'زيارة', 'س', 'ساء', 'سابع', 'سادس', 'سبت', 'سبتمبر', 'سبحان', 'سبع', 'سبعة', 'سبعمئة', 'سبعمائة', 'سبعون', 'سبعين', 'ست', 'ستة', 'ستكون', 'ستمئة', 'ستمائة', 'ستون', 'ستين', 'سحقا', 'سرا', 'سرعان', 'سقى', 'سمعا', 'سنة', 'سنتيم', 'سنوات', 'سوف', 'سوى', 'سين', 'ش', 'شباط', 'شبه', 'شتانَ', 'شخصا', 'شرع', 'شمال', 'شيكل', 'شين', 'شَتَّانَ', 'ص', 'صاد', 'صار', 'صباح', 'صبر', 'صبرا', 'صدقا', 'صراحة', 'صفر', 'صهٍ', 'صهْ', 'ض', 'ضاد', 'ضحوة', 'ضد', 'ضمن', 'ط', 'طاء', 'طاق', 'طالما', 'طرا', 'طفق', 'طَق', 'ظ', 'ظاء', 'ظل', 'ظلّ', 'ظنَّ', 'ع', 'عاد', 'عاشر', 'عام', 'عاما', 'عامة', 'عجبا', 'عدا', 'عدة', 'عدد', 'عدم', 'عدَّ', 'عسى', 'عشر', 'عشرة', 'عشرون', 'عشرين', 'عل', 'علق', 'علم', 'على', 'علي', 'عليك', 'عليه', 'عليها', 'علًّ', 'عن', 'عند', 'عندما', 'عنه', 'عنها', 'عوض', 'عيانا', 'عين', 'عَدَسْ', 'غ', 'غادر', 'غالبا', 'غدا', 'غداة', 'غير', 'غين', 'ـ', 'ف', 'فإن', 'فاء', 'فان', 'فانه', 'فبراير', 'فرادى', 'فضلا', 'فقد', 'فقط', 'فكان', 'فلان', 'فلس', 'فهو', 'فو', 'فوق', 'فى', 'في', 'فيفري', 'فيه', 'فيها', 'ق', 'قاطبة', 'قاف', 'قال', 'قام', 'قبل', 'قد', 'قرش', 'قطّ', 'قلما', 'قوة', 'ك', 'كأن', 'كأنّ', 'كأيّ', 'كأيّن', 'كاد', 'كاف', 'كان', 'كانت', 'كانون', 'كثيرا', 'كذا', 'كذلك', 'كرب', 'كسا', 'كل', 'كلتا', 'كلم', 'كلَّا', 'كلّما', 'كم', 'كما', 'كن', 'كى', 'كيت', 'كيف', 'كيفما', 'كِخ', 'ل', 'لأن', 'لا', 'لا سيما', 'لات', 'لازال', 'لاسيما', 'لام', 'لايزال', 'لبيك', 'لدن', 'لدى', 'لدي', 'لذلك', 'لعل', 'لعلَّ', 'لعمر', 'لقاء', 'لكن', 'لكنه', 'لكنَّ', 'للامم', 'لم', 'لما', 'لمّا', 'لن', 'له', 'لها', 'لهذا', 'لهم', 'لو', 'لوكالة', 'لولا', 'لوما', 'ليت', 'ليرة', 'ليس', 'ليسب', 'م', 'مئة', 'مئتان', 'ما', 'ما أفعله', 'ما انفك', 'ما برح', 'مائة', 'ماانفك', 'مابرح', 'مادام', 'ماذا', 'مارس', 'مازال', 'مافتئ', 'ماي', 'مايزال', 'مايو', 'متى', 'مثل', 'مذ', 'مرّة', 'مساء', 'مع', 'معاذ', 'معه', 'معها', 'مقابل', 'مكانكم', 'مكانكما', 'مكانكنّ', 'مكانَك', 'مليار', 'مليم', 'مليون', 'مما', 'من', 'منذ', 'منه', 'منها', 'مه', 'مهما', 'ميم', 'ن', 'نا', 'نبَّا', 'نحن', 'نحو', 'نعم', 'نفس', 'نفسه', 'نهاية', 'نوفمبر', 'نون', 'نيسان', 'نيف', 'نَخْ', 'نَّ', 'ه', 'هؤلاء', 'ها', 'هاء', 'هاكَ', 'هبّ', 'هذا', 'هذه', 'هل', 'هللة', 'هلم', 'هلّا', 'هم', 'هما', 'همزة', 'هن', 'هنا', 'هناك', 'هنالك', 'هو', 'هي', 'هيا', 'هيهات', 'هيّا', 'هَؤلاء', 'هَاتانِ', 'هَاتَيْنِ', 'هَاتِه', 'هَاتِي', 'هَجْ', 'هَذا', 'هَذانِ', 'هَذَيْنِ', 'هَذِه', 'هَذِي', 'هَيْهات', 'و', 'و6', 'وأبو', 'وأن', 'وا', 'واحد', 'واضاف', 'واضافت', 'واكد', 'والتي', 'والذي', 'وان', 'واهاً', 'واو', 'واوضح', 'وبين', 'وثي', 'وجد', 'وراءَك', 'ورد', 'وعلى', 'وفي', 'وقال', 'وقالت', 'وقد', 'وقف', 'وكان', 'وكانت', 'ولا', 'ولايزال', 'ولكن', 'ولم', 'وله', 'وليس', 'ومع', 'ومن', 'وهب', 'وهذا', 'وهو', 'وهي', 'وَيْ', 'وُشْكَانَ', 'ى', 'ي', 'ياء', 'يفعلان', 'يفعلون', 'يكون', 'يلي', 'يمكن', 'يمين', 'ين', 'يناير', 'يوان', 'يورو', 'يوليو', 'يوم', 'يونيو', 'ّأيّان']


[words.remove(w) for w in words if w in Arabic_stop_words] 


#---------------------------

# sort words
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))

#              #  print #               #

""" # print corpus size 
print (len(corpus), "corpus")
# print classes size with classes = intentsa
print (len(classes), "classes", classes)
# print words size = all lemmatized words, vocabulary
print (len(words), "unique lemmatized words", words) """

# to save time instead of biulding them each time we open the GUI
pickle.dump(words,open('wordsa.pkl','wb'))
pickle.dump(classes,open('classesa.pkl','wb'))

# -------------- Create training and testing data ------------------- 
#                 creating the Bag of Words model

# Splitting the intents_a into the Training set and Test set
# input = the pattern, output = the class our input pattern belongs to 
# But the computer doesn’t understand text, so we will convert text into numbers:


# create the training data
training = []
# create an empty array for the output that matches the size of classes tags
array_empty = [0] * len(classes)

# training set, bag of words for each sentence
for cor in corpus: # tokenized words with thier tags: ['أهلا','مرحبا']greeting
    # initialize the bag of words 
    bag = [] 
    # list of tokenized words for the pattern [0]['أهلا','مرحبا'] 
    pattern_words = cor[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(array_empty)
    output_row[classes.index(cor[1])] = 1 # cor[1] = greeting
    
    # create the training data(0's and 1's) with the [bag of words][classes]
    training.append([bag, output_row]) 
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intentsa, classes
train_x = list(training[:,0]) # all rows, first col 
train_y = list(training[:,1]) # all rows, 2nd col, which pattern belongs to which class
print("Training data created")


# -------------- Bilding the model -------------------



# Define model architecture, linear stack of layers
model = Sequential()

# Create the model
# 3 layers. First layer 128 neurons, second layer 64 neurons 
# 3rd output layer contains number of neurons
# equal to number of intentsa to predict output intent with softmax
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) #train_x = size
model.add(Dropout(0.5)) # to control overfitting problem
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax')) # which pattern belongs to which class

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) #learning rate =0.01
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodela.h5', hist) # to save time instead of training the model each time