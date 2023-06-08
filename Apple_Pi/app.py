import streamlit as st
import numpy as np
import pandas as pd
from cmfrec import CMF
from PIL import Image

# Load the ratings data
ratings = pd.read_csv("D:/UM/year 2 sem 2/Computing Mathematics 2/Apple_Pi/df_rearranged.csv")

# Load the app names data
app_names = pd.read_csv("D:/UM/year 2 sem 2/Computing Mathematics 2/Apple_Pi/df_app_names.csv")

# Fit the collaborative filtering model
model = CMF(method="als", k=26, lambda_=1e+1)
model.fit(ratings)

# Define the Streamlit app
def main():
    # Set the page title
    st.title("Student Productivity App Recommendation System")

    # Retrieve the user ratings for each app
    user_ratings = []  # Store the user ratings here

    # Display the app pictures and rating stars
    for app_id in range(26):
        # Print the app name
        app_name = app_names.AppName.loc[app_names.AppID == app_id+1].values[0]
        st.write(f"- {app_name}")
        
        # Display the app picture
        st.image(Image.open(f"D:/UM/year 2 sem 2/Computing Mathematics 2/Apple_Pi/app_images/app{app_id + 1}.png"), width=100)

        # Display the rating stars
        rating = st.slider("Rate the app", 0, 5, key=app_id+1)
        

        # Store the user rating for the app
        user_ratings.append(rating)
        

    if st.button("Generate Recommendations"):
        # Make app recommendations based on user ratings
        recommended_apps = model.topN(np.array(user_ratings), n=5)

        # Display the recommended app pictures
        st.subheader("Recommended Apps")
        for app_id in recommended_apps:
            print(app_id)
            app_name = app_names.AppName.loc[app_names.AppID == app_id].values[0]
            st.write(f"- {app_name}")
            st.image(Image.open(f"D:/UM/year 2 sem 2/Computing Mathematics 2/Apple_Pi/app_images/app{app_id}.png"), width=100)

if __name__ == "__main__":
    main()
