<h2 align="center"> Sentiments-Classifier-using-BERT</h2>

<h3 align="left"> IMDB - Movie Reviews-using-BERT</h3>

 <p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/74568334/140315545-de6e357e-a0fa-43e2-891f-44afe6baf2ba.png">
</p> 

<h3 align="left">Introduction </h3>
 
 
<p style= 'text-align: justify;'> The process of computationally identifying and categorizing opinions expressed in a piece of text, especially to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral. Understanding people’s emotions is essential for businesses since customers are able to express their thoughts and feelings more openly than ever before. By automatically analysing customer feedback, from survey responses to social media conversations, brands are able to listen attentively to their customers, and tailor products and services to meet their needs.</p>
  

<p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/74568334/140315314-aec1185c-fc9a-4657-bb9f-372467743f7e.jpg">
</p> 

<h2 align="center"> Technologies Used </h2>
 
 ```
 1. IDE - Pycharm
 2. BERT - Pre-Trained Model
 3. GPU - P-4000
 4. Google Colab - Text Analysis
 5. Flask- Fast API
 6. Postman - API Tester
 7. TensorFlow - Hub - Convert to the Tokenizer
 ```
 
<p style= 'text-align: justify;'> 
 
   🔑 Prerequisites
      All the dependencies and required libraries are included in the file requirements.txt

      Python 3.6
 
</p>

<h2 align="center"> Dataset </h2>

<p style= 'text-align: justify;'> IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms. </p>

For Dataset Please click [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/version/1)


<h2 align="center"> Process - Flow of This project </h2>

<p align="center">
  <img width="1000" src="https://user-images.githubusercontent.com/74568334/140325153-43b7c157-f8be-4c81-a37a-b780816e55bc.jpeg">
</p> 

<h2 align="center"> 🚀 Installation For  Application Sentiments-Classifier-using-BERT </h2>

1. Clone the repo

```
git clone https://github.com/KrishArul26/Sentiments-Classifier-using-BERT.git

```
2. Change your directory to the cloned repo

```
cd Sentiments-Classifier-using-BERT

```
3. Create a Python 3.6 version of virtual environment name 'bert' and activate it

```
pip install virtualenv

virtualenv bert

bert\Scripts\activate

```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required!!!

```
pip install -r requirements.txt

```

<h2 align="center"> 💡 Working </h2>

Type the following command:

```
python app.py

```
After that You will see the running IP adress just copy and paste into you browser and import or upload your speech then click the predict button.


<h2 align="center"> Folder Structure </h2>

<h4 align="left"> In this section, contains the project directory, explanation of each python file presents in the directory.  </h2>


<h3 align="left">1. Project Directory</h3>


<h4 align="left"> Below picture illustrate the complete folder structure of this project.</h4>


<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140577934-92f60e0d-c905-478e-be62-638bd6a7ad82.png">
</p> 


<h3 align="left">2. preprocess.py </h3>

<p style= 'text-align: justify;'> Below picture illustrate the preprocess.py file, It does the necessary text cleaning process such as removing punctuation, numbers, lemmatization. And it will create train_preprocessed, validation_preprocessed and test_preprocessed pickle files for the further analysis.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140578710-2b346932-32c8-4f60-b9bf-b79fbb4fbf10.png">
</p> 

<h3 align="left">3. word_embedder_gensim.py </h3>

<p style= 'text-align: justify;'> Below picture illustrate the word_embedder_gensim.py, After done with text pre-processing, this file will take those cleaned text as input and will be creating the Word2vec embedding for each word.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140579065-79a7e215-1f8f-4715-816c-0247d007a520.png">
</p> 


<h3 align="left">4. rnn_w2v.py </h3>

<p style= 'text-align: justify;'>Below picture illustrate the rnn_w2v.py, After done with creating Word2vec for each word then those vectors will use as input for creating the LSTM model and Train the LSTM (RNN) model with body and Classes. </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140579999-d0ae2ac4-74bc-460d-82eb-3ee7cbb40a73.png">
</p> 

<h3 align="left">5. index.htmml </h3>

<p style= 'text-align: justify;'>Below picture illustrate the index.html file, these files use to create the web frame for us. </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140581823-b9f8a43a-e317-4e18-b895-0983d905cb60.png">
  <img width="600" src="https://user-images.githubusercontent.com/74568334/140581821-30aca256-6442-4b0e-8e29-9bef67f2d118.png">
 
</p> 


<h3 align="left">6. main.py </h3>
<p style= 'text-align: justify;'> Below picture illustrate the main.py, After evaluating the LSTM model, This files will create the Rest -API, To that It will use FLASK frameworks and get the request from the customer or client then It will Post into the prediction files and Answer will be deliver over the web browser.   </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140581040-86b02b9a-fb8c-4f10-9ebf-03e05573f7a6.png">
 
</p> 

<h2 align="center"> 🔑 Results </h2>



<h4 align="Left"> Result - 1 </h4>

 <p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/74568334/140320085-08d15e68-fa17-4207-aff9-d751ab51d1ef.png">
</p>



<h4 align="Left"> Result - 2 </h4>

<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/74568334/140329890-72d738d9-6838-4235-9c0b-86f6673bbbd2.png">
</p>

<h2 align="center"> Conclusion </h2>

<p style= 'text-align: justify;'> BERT pre-trained model is the best attention model for the NLP. Here, BERT works well with text for sentiment analysis but Its model weight is a little bigger than other NLP models.</p>



