import streamlit as st
from gait_analysis import GaitAnalysis
import os
import uuid
import pandas as pd  # Import pandas for creating the summary dataframe
import streamlit.components.v1 as components


# Display UI using Streamlit
class StreamlitApp:
    def __init__(self):

        st.set_page_config(layout="wide")  # Use a wider layout for better spacing
        st.title("AI-Powered Gait Analyzer with Computer Vision")

        st.caption(
            """
            Analyze your gait with Computer Vision. This tool provides key metrics based on a video of your movement.
            """
        )

        st.header("1. Enter Your Details")
        # Use columns for a cleaner layout
        col1, col2, col3 = st.columns(3)

        with col1:
            height = st.number_input("Height (cm)", min_value=1.0, value=170.0, step=1.0,
                                     help="Enter your height in centimeters.")

        with col2:
            weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0, step=1.0,
                                     help="Enter your weight in kilograms.")

        with col3:
            distance = st.number_input("Distance Covered (m)", min_value=0.1, value=5.0, step=0.1,
                                       help="Enter the distance you walked in the video in meters.")

        st.header("2. Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a short video of you moving from left to right (or) right to left covering your entire body. Preferably Side View angle.",
            type=["mp4", "mov"])

        if uploaded_file is not None:
            # Add a button to trigger the analysis
            if st.button("Analyze Gait", type="primary"):
                input_directory = "input_videos"
                if not os.path.exists(input_directory):
                    os.makedirs(input_directory)
                input_video_filename = uuid.uuid4().hex + ".mp4"
                video_path = os.path.join(input_directory, input_video_filename)

                with open(video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Use st.spinner to show a loading message during processing
                with st.spinner('🏃‍♂️ Analyzing your gait... Please wait.'):
                    # IMPORTANT: Your GaitAnalysis class must be updated to accept these arguments
                    # and your process_video method must return speed_mps
                    gait_analysis = GaitAnalysis(
                        video_path=video_path,
                        height=height,
                        weight=weight,
                        distance=distance
                    )
                    # The new parameter `speed_mps` is now returned
                    output_video, df, result, plt, distance_df, speed_mps , step_count = gait_analysis.process_video()

                st.success('✅ Analysis complete!')



                # --- End of New Changes ---

                st.header("Annotated video:")
                video_file = open(output_video, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes, format="video/webm", start_time=0)

                # --- Start of New Changes ---

                st.header("Analysis Summary")

                # Use columns to display key metrics in a card-like format
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

                with metric_col1:
                    st.metric(label="Height", value=f"{height} cm")

                with metric_col2:
                    st.metric(label="Weight", value=f"{weight} kg")

                with metric_col3:
                    st.metric(label="Distance Covered", value=f"{distance} m")

                with metric_col4:
                    # Format the speed to 2 decimal places for better readability
                    st.metric(label="Average Speed", value=f"{speed_mps:.2f} m/s")

                with metric_col5:
                    # Format the speed to 2 decimal places for better readability
                    st.metric(label="Step Count", value=f"{step_count} step")

                st.markdown("---")  # Add a horizontal line for separation

                st.header("Plotting the Distances, Peaks and Minima ")

                st.subheader("Gait Analysis Both Leg: ")
                st.pyplot(plt.figure(1), clear_figure=True)

                st.header("Gait Data Time Analysis:")
                st.dataframe(df)

                st.header("Gait Data Distance Analysis:")
                st.dataframe(distance_df)

                csv = self.convert_df(df)

                st.download_button(
                    "Download Gait Data",
                    csv,
                    "gait_analysis.csv",
                    "text/csv",
                    key='download-csv'
                )

    # Download the dataframe as a .csv file
    @staticmethod
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


if __name__ == "__main__":
    app = StreamlitApp()
