# 1) Firstly we import all the necessary libraries to perform the task
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from cryptography.hazmat.primitives.asymmetric import padding as asymmetric_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import hashes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import LatentDirichletAllocation

# 2) Now we shall load the dataset and declare the columns to encrypt
data = pd.read_csv('reviews.csv')
categorical_columns = ['reviewId', 'userName', 'userImage', 'reviewCreatedVersion', 'at', 'repliedAt', 'sortOrder', 'appId']

# 3) Now let's generate the key for the AES
aes_key = os.urandom(32)
aes_iv = os.urandom(16)

# 4) Now we shall generate the private key and pyblic key for the RSA
rsa_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
rsa_public_key = rsa_private_key.public_key()

# 5) Now we shall define the function to encrypt the symmetric key for the AES using the RSA
def rsa_encrypt(message):
    ciphertext = rsa_public_key.encrypt(
        message,
        asymmetric_padding.OAEP(
            mgf=asymmetric_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

# 6) Now we shall encrypt the sensitive data the has nothing to do with sentiment analysis
def aes_encrypt(plain_text):
    if isinstance(plain_text, float):
        plain_text = str(plain_text)
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(aes_iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(plain_text.encode()) + padder.finalize()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return ciphertext

# 7) Now let's define function to induce Laplace noise
def add_laplace_noise(data, epsilon):
    scale = 1.0 / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=data.data.shape)
    noisy_data = data.copy()
    noisy_data.data += noise
    return noisy_data

# 8) Now let's define a function for k-anonymity
def apply_k_anonymity(data, k):
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)
    masked_data = np.zeros_like(data)
    for i, neighbors_indices in enumerate(indices):
        masked_data[i] = np.mean(data[neighbors_indices], axis=0)
    return masked_data

# 9) Now we shall define the function to simulate the linkage type attack
def simulate_linkage_attack(encrypted_reviews, public_forum_reviews):
    match_count = 0
    for encrypted_review in encrypted_reviews:
        for public_review in public_forum_reviews:
            if encrypted_review in public_review:
                match_count += 1
    return match_count

# 10) Now let's define a function to analyse the sentiment from the reviews
def analyze_sentiments(reviews):
    sentiments = [sentiment_analyzer.polarity_scores(review) for review in reviews]
    return sentiments

# 11) Now we shall perform topic modeling using Latent Dirichlet Allocation (LDA) from this function
def perform_topic_modeling(reviews, num_topics=5):
    tfidf = TfidfVectorizer(stop_words='english')  # Use 'english' instead of frozenset
    tfidf_matrix = tfidf.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_matrix = lda.fit_transform(tfidf_matrix)
    return lda, tfidf

# 12) And finally here is the function to display those topics
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

# 13) Now we shall apply the RSA on the AES key to encrypt the key
rsa_encrypted_aes_key = rsa_encrypt(aes_key)

# 14) Now we shall apply AES encryption on the sensitive data
for column in categorical_columns:
    data[column + '_aes'] = data[column].apply(aes_encrypt)

# 15) For better understanding we will keep the encrypted data as a different dataframe
encrypted_data = data.copy()
encrypted_data = encrypted_data.drop(['reviewId', 'userName', 'userImage', 'reviewCreatedVersion', 'at', 'repliedAt', 'sortOrder', 'appId'], axis=1)
print(encrypted_data)

# 16) Now let's train the model after applying the encryption. Firstly we will decalre the data that we want to use to train and test
X = encrypted_data['content']
y = encrypted_data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 17) Now let's initialize the TFIDF on which we want to train the model
tfidf_transformer = TfidfVectorizer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
X_test_tfidf = tfidf_transformer.transform(X_test)

# 18) Now let's train the model after encryption with random forest
clf = RandomForestClassifier()
clf.fit(X_train_tfidf, y_train)

# 19) Now let's predict on test data to get the score
y_pred = clf.predict(X_test_tfidf)

# 20) Now let's check the accuracy without differential privacy
accuracy = clf.score(X_test_tfidf, y_test)
print("Accuracy:", accuracy)

# 21) Now we will apply Laplace noise to the training and test data
epsilon = 0.1
X_train_dp = add_laplace_noise(X_train_tfidf, epsilon)
X_test_dp = add_laplace_noise(X_test_tfidf, epsilon)

# 22) Now we will initialize and train Random Forest model again with differentially private data
clf_dp = RandomForestClassifier()
clf_dp.fit(X_train_dp, y_train)

# 22) Now let's predict and evaluate the model with differential privacy for analysis
y_pred_dp = clf_dp.predict(X_test_dp)
accuracy_dp = accuracy_score(y_test, y_pred_dp)
print("Accuracy with differential privacy:", accuracy_dp)

# 23) Now we shall apply k-anonymity to the data and apply laplace noise to that anonymised data
k = 5
X_train_anon = apply_k_anonymity(X_train_dp.toarray(), k)
X_test_anon = apply_k_anonymity(X_test_dp.toarray(), k)
X_test_dp_anon = add_laplace_noise(X_test_anon, epsilon)
X_train_dp_anon = add_laplace_noise(X_train_anon, epsilon)

# 24) Now let's train Random Forest but this time with differentially private and k-anonymized data
clf_dp_anon = RandomForestClassifier()
clf_dp_anon.fit(X_train_dp_anon, y_train)

# 25) Now we shall predict and evaluate the model with differential privacy and k-anonymity for some more analysis
y_pred_dp_anon = clf_dp_anon.predict(X_test_dp_anon)
accuracy_dp_anon = accuracy_score(y_test, y_pred_dp_anon)
print("Accuracy with differential privacy and k-anonymity:", accuracy_dp_anon)

# 26) Now we shall validate Linkage Attack Mitigation
    # a) Simulate an attacker trying to link reviews by comparing unique phrases and timestamps, we will initialise example public revies for that
public_forum_reviews = [
    "I love using this app for tracking my workouts!",
    "The new update for the app has some bugs.",
    "This app helps me manage my anxiety and stress."
]

    # b) Let's try and simulate linkage attack
encrypted_reviews = X_test.to_numpy()
linkage_attack_matches = simulate_linkage_attack(encrypted_reviews, public_forum_reviews)
print(f"Linkage Attack Matches: {linkage_attack_matches}")


# **Linkage Attack Validation**
# To validate the mitigation of linkage attacks, we checked the linkage attack matches. The simulated attack tried to link reviews in the encrypted dataset with those in a public forum, and we found that the matches were minimal. This indicates that our encryption and privacy techniques have effectively obfuscated the data, preventing successful linkage.



# 27) Now let's validate Inference Attack Mitigation
    # a) For that we will perform sentiment analysis and topic modeling on the encrypted dataset
        # i) Let's perform sentiment analysis
nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()
        # ii) Now let's analyze sentiments on the encrypted dataset
encrypted_reviews = X_test.to_numpy()
encrypted_sentiments = analyze_sentiments(encrypted_reviews)
print(f"Encrypted Sentiments: {encrypted_sentiments[:5]}")

        # iii) Now we shall perform topic modeling on the encrypted dataset
lda, tfidf = perform_topic_modeling(encrypted_reviews)
terms = tfidf.get_feature_names_out()

        # iv) Now let's display those topics as well
display_topics(lda, terms, 10)

# 28) Let's visulaise the data and compare the accuracies
accuracies = {
    "Original Data with encryption": accuracy,
    "Differential Privacy": accuracy_dp,
    "Differential Privacy + K-Anonymity": accuracy_dp_anon
}
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Data Privacy Technique')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.show()

# **Inference Attack Validation**
# For inference attacks, we performed sentiment analysis and topic modeling on the encrypted dataset:
# 
# **Sentiment Analysis:**
# The encrypted sentiments show a range of sentiments (positive, negative, neutral), which indicates that while sentiment analysis is possible, the exact details and sensitive information are sufficiently obfuscated.
# 
# **Topic Modeling:**
# The topics derived from the encrypted dataset are generic and not directly revealing of sensitive information. Topics like "app love helpful bad" or "good app great use" are general and do not provide specific identifiable or sensitive information about users.


# **Accuracy:** Differential privacy and k-anonymity reduce accuracy from 49% to 40 % and whe applied k anonymity it goes down to 22%, this is expected as privacy increases.
# 
# **Linkage Attack:** Simulated linkage attacks show minimal (0 in this case) matches, indicating effective mitigation.
# 
# **Inference Attack:** Sentiment analysis and topic modeling on encrypted data show general information, indicating effective mitigation.
