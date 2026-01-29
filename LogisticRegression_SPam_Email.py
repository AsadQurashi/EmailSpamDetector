import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , f1_score , confusion_matrix , precision_score , roc_auc_score
import os

class DataLoader():
    def __init__(self , filename = 'spam.csv'):
        self.df = None
        self.filename = filename
    
    def load_data(self):
        # Load data from csv file
        try:
            self.df = pd.read_csv(self.filename)
            print(self.df.head())
            print("Data loaded successfully")
            return True
        except FileNotFoundError:
            print("Data is not loaded : ",self.filename)
            return False
            
    
    def prepareData(self):
        try:
            if 'v1' in self.df.columns and 'v2' in self.df.columns:
                self.df = self.df.rename({'v1' : 'Label' , 'v2' : 'Text'})
                print("Name has been renamed")
            else:
                print("Name is already well managed")

            # Converting label into ham(0) and Spam(1)
            self.df['Category'] = self.df['Category'].map({'ham' : 0 , 'spam' : 1})
            # Remove missing values
            self.df = self.df.dropna()

            # 
            print(self.df.head())
            # EDA
            print("=="*50)
            print("Data Info")
            print("Total Email :",len(self.df))
            print("Spam Emails :",self.df['Category'].sum())
            print("Ham Emails :",self.df['Category'].sum())
            return self.df[['Message', 'Category']]

        except Exception as e:
            print(f"Error in prepareData: {e}")
            import traceback
            traceback.print_exc()
            return None

class SpamDetector():

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = None
        
    def train_model(self , texts , labels):
        self.vectorizer = CountVectorizer(
            max_features= 1000,
            stop_words='english'
            )
        
        # Convert text to numbers
        x = self.vectorizer.fit_transform(texts)

        # Split into train and test
        x_train , x_test , y_train , y_test = train_test_split(x , labels , test_size=0.2 , random_state=41)

        # Create and train model
        self.model = LogisticRegression(penalty='l2' , solver='lbfgs' , C=1.0 , max_iter=1000)
        self.model.fit(x_train , y_train)

        # Test the model
        y_pred = self.model.predict(x_test)
        acc = accuracy_score(y_test , y_pred)

        self.is_trained = True
        print("Model is trained")
        return acc
    
    def test_model(self , x_test , y_test):
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test , y_pred)

        print("f\nModel Performance")
        print(f"Accuracy : {accuracy:.1f}")
        print(f"Accuracy out of 100 {int(accuracy * 100)}")

        return accuracy
    
    def Predict(self , email_text):
        if not self.is_trained:
            return("Model is not trained")
        
        # convert email into numbers
        email_converter = self.vectorizer.transform([email_text])

        # prediction
        probability = self.model.predict_proba(email_converter)[0,1]

        # Make desicion
        is_spam = probability > 0.5

        return{
            'email_text' : email_text[:50] + ('...' if len(email_text) > 50 else ''),
            'is_spam' : is_spam,
            'probability' : probability,
            'label' : ': Spam' if is_spam else ': Ham',
            'confidence' : self.get_confidence(probability)
        }
    
    def get_confidence(self , prob):
        """Returns confidence level based on probability"""
        if prob > 0.9 or prob < 0.1:
            return 'Very High'
        elif prob > 0.7 or prob < 0.3:
            return 'High'
        else:
            return 'Medium'
    
    def show_top_words(self , n=10):
        if not self.is_trained:
            return ("Model is not trained")
        
        features = self.vectorizer.get_feature_names_out()
        coefficents = self.model.coef_[0]

        # Spam indicators
        spam_indicators = sorted(zip(features , coefficents) , key=lambda x : x[1] , reverse=True)

        for words , scors in spam_indicators:
            print(f"Words : {words} , Score : {scors:.3f}")
    
def run_test(detector):
    test_email = [
    "WIN FREE PRIZE! Call now to claim",
    "Meeting at 3 PM tomorrow in room 5",
    "URGENT: Your account needs verification",
    "Hi mom, what's for dinner tonight?",
    "EARN MONEY FAST working from home!!!",
    "Project deadline extended to next week",
    "FREE BITCOINS!!! Double your money now!",
    "Lunch tomorrow? I found a new restaurant",
    ]

    print("\n" + "="*50)
    print("TESTING THE DETECTOR")
    print("="*50)

    for email in test_email:
        result = detector.Predict(email)
        print(f"\n{result['email_text']}")
        print(f"   Result: {result['label']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Probability: {result['probability']:.1%}")

def interactive_mode(detector):
    "Let user test their email"
    print("\n" + "="*50)
    print("🎮 INTERACTIVE MODE")
    print("="*50)
    print("Type your emails and see if they're spam!")
    print("Type 'quit' to exit\n")

    while True:
        user_email = input("Enter your email\n")

        if user_email.lower == 'quit':
            break

        result = detector.Predict(user_email)

        print(f"\n🔍 Analysis:")
        print(f"   Result: {result['label']}")
        print(f"   Confidence: {result['confidence']}")

        if result['probability'] > 0.7:
            print("I'm very confident!")
        elif result['probability'] > 0.5:
            print("I think so, but not certain...")
        else:
            print("Looks safe to open!")
        
        print("-"*30)

def main():
    """Main function that runs the whole program"""
    print("="*50)
    print("📧 SIMPLE SPAM DETECTOR")
    print("="*50)

    print("\n1. Loading Data...")
    Base_url = os.getcwd()
    url = os.path.join(Base_url ,'../','DataSet/SpamEmail' , 'spam.csv')
    loader = DataLoader(url)

    if not loader.load_data():
        return
    data = loader.prepareData()

    # Step 2: Create and train detector
    print("\n2. Creating spam detector")
    detector = SpamDetector()
    accuracy = detector.train_model(data['Message'] , data['Category'])
    detector.show_top_words(8)
    run_test(detector)
    interactive_mode(detector)
    print("\n" + "="*50)
    print("✅ Program complete!")
    print(f"   Final accuracy: {accuracy:.1%}")
    print("="*50)

if __name__ == '__main__':
    main()


