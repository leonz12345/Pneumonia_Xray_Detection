# Pneumonia X-ray Detection

## What We Did
With the continuous outbreak of COVID-19 around the world, many health experts believe that the pandemic will not end in 2021, and the cease of the pandemic will likely involve a mix of efforts. The main objective of this report is to provide a comprehensive analysis and concrete data-driven recommendations on the devastating long-term economic implications of COVID-19. To define economic impact, we found three important economic metrics: GDP, unemployment rate, and stock market index. We used principal component analysis to aggregate these three metrics into a single score which we call Economic Develop Index, EDI. We then build a regression model to see which attributes are significant. Finally, we used time-series forecasting to predict how a country would perform in the foreseeable future. Our model shows increases in Population Density, Death Rate, New Deaths Per Million, Stringency Index, and Hospital Beds Per Thousand will decrease the overall EDI. The findings match our expectation because of the increase of death rate, stringency index, new death counts, and hospital beds indicate the increase of severity for COVID-19. Moreover, we used the United States to demonstrate how our findings could be used to predict the EDI economic indicator. The same approach could be applied to other countries/cities with appropriately processed data. Finally, we made recommendations to countries facing the devastating long-term economic implications of this pandemic.

## Our Approach
![Model Selection](https://github.com/leonz12345/Pneumonia_Xray_Detection/blob/main/Static/model_selection.png)
![Webapp](https://github.com/leonz12345/Pneumonia_Xray_Detection/blob/main/Static/webapp_inferface.jpg)

## Methods Used
- Convolutionol neural network
- Image augmentation
- Grid search

## Tools Used
- Pandas, NumPy, OpenCV for image data processing
- Keras for data augmentation
- Tensorflow, Keras for building effective models
- Streamlit for deploying model into web application
