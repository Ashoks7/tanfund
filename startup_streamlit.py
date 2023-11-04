import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import datetime as dt

#2. upload the file

startup_df = pd.read_csv("C:\Users\DELL\Downloads\archive (16)")
print(startup_df.head(5))