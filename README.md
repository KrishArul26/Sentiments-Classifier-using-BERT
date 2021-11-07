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


<h4 align="left"> Below picture illustrate the complete folder structure of this project. This folder will keep the model that have been trained on the
 dataset using BERT architecture.</h4>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140644137-c4f60ebf-a10f-4830-8495-06f645aedc1f.png">
</p> 

<h3 align="left">2. train_bert_model.py </h3>

<p style= 'text-align: justify;'> The following image illustrates the file train_model_bert.py. It does the necessary text cleanup, such as removing punctuation and numbers. And it creates tokenizers from the TnsorFlo - Hub Bert model. These tokenizers are padded according to the specified length. Finally, the BERT model is trained using the train dataset.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140644134-907b1eae-6f4f-4389-898c-9a2072177ab8.png">
</p> 

<h3 align="left">3. prediction File.py </h3>

<p style= 'text-align: justify;'> Below picture illustrate the prediction File.py, After done with train the BERT model, This file processes the test data in the same way as the training data and will predict the test data.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140644141-98fdb449-401c-4904-872c-0239ba291e8b.png">
</p> 


<h3 align="left">5. index.htmml </h3>

<p style= 'text-align: justify;'>Below picture illustrate the index.html file, these files use to create the web frame for us. </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140644139-effba2f6-9217-4a5a-ab48-e19f2bb309f6.png"
 
</p> 


<h3 align="left">6. main.py </h3>
<p style= 'text-align: justify;'> The following image illustrates picture illustrate the main.py. After evaluating the model BERT, this file creates the rest - API. To do this, it uses FLASK frameworks and receives the request from the client. Then it is posted to the prediction files and the response is delivered through the web browser. </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140644140-f5769f83-a48f-44d5-a7a0-f9539d0ded5c.png">
 
</p> 

<h2 align="center"> 🔑 Results </h2>



<h4 align="Left"> Result - 1 </h4>

 <p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/74568334/140644146-d71a9b20-8370-4798-b972-df94da0fe3b7.png">
</p>


<h2 align="center"> Conclusion </h2>

<p style= 'text-align: justify;'> The pre-trained model BERT is the best attention model for NLP task. Here BERT works well with text for sentiment analysis, but its model weight is somewhat greater than that of other NLP models.</p>



