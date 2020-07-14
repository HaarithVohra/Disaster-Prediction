import re, unicodedata
import nltk
import inflect
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessing:
    def __init__(self, data):
        """Preprocess the data"""
        self.sample, self.words = None, None
        self.data = data
        for i in tqdm(range(len(data))):
            self.sample = data["text"][i]
            self.data["text"][i] = self.preprocess()
        self.data.drop(self.data[self.data.text == ""].index, inplace=True) # delete empty rows
        self.data.drop_duplicates(subset="text", keep=False, inplace=True) # dropping duplicate values 
        self.data.reset_index(inplace=True, drop=True) # reset the indices
    
    def to_csv_file(self, file_name):
        """Save the data in a file whose name is specified above"""
        self.data.to_csv("../Data/" + file_name + ".csv", index=None, header=True)
    
    def remove_at_user_link(self):
        """Remove @user_link from a tweet"""
        self.sample = re.sub(r"@[^\s]+", "", self.sample)
    
    def remove_URL(self):
        """Remove URLs from a sample string"""
        self.sample = re.sub(r"http\S+", "", self.sample)
        
    def remove_short_words(self):
        """Remove letters with length <=3"""
        self.sample = re.sub(r"\b\w{1,3}\b", "", self.sample)
    
    def remove_non_ascii(self):
        """Remove non-ASCII characters from list of tokenized words"""
        self.words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in self.words]
    
    def to_lowercase(self):
        """Convert all characters to lowercase from list of tokenized words"""
        self.words = [word.lower() for word in self.words]
    
    def remove_punctuation(self):
        """Remove punctuation from list of tokenized words"""
        self.words = [clean for clean in (re.sub(r'[^\w\s]', '', word) for word in self.words) if clean != '']
    
    def replace_numbers(self):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        self.words = [inflect.engine().number_to_words(word) if word.isdigit() else word for word in self.words]
    
    def delete_numbers(self):
        """Delete all the numbers"""
        self.words = [word for word in self.words if not word.isdigit()]
    
    def remove_stopwords(self):
        """Remove stop words from list of tokenized words"""
        self.words = [word for word in self.words if word not in stopwords.words('english')]
    
    def stem_words(self):
        """Stem words in list of tokenized words"""
        stemmer = PorterStemmer()
        self.words = [stemmer.stem(word) for word in self.words]
    
    def lemmatize_verbs(self):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        self.words = [lemmatizer.lemmatize(word, pos='v') for word in self.words]
    
    def clean(self):
        """Clean the text before tokenization"""
        self.remove_at_user_link()
        self.remove_URL()
    
    def normalize(self):
        """Normalize the text using common preprocessing techniques"""
        self.remove_non_ascii()
        self.to_lowercase()
        self.remove_punctuation()
        self.delete_numbers()
        self.remove_stopwords()
        self.remove_short_words()
        self.lemmatize_verbs() # self.stem_words()
    
    def preprocess(self):
        """Clean and normalize text"""
        self.clean()
        self.words = nltk.word_tokenize(self.sample)
        self.normalize()
        return ' '.join(self.words)