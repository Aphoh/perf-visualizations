from functools import lru_cache
import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import numpy as np

num_experts = st.slider("num_experts", 1, 256)
if num_experts > 1:
    K = st.slider("K", 1, num_experts)
hidden_dim = st.slider("Hidden Dimension", 512, 16384, step=512)
ff_dim_mult = st.slider("FF Dimension", 1, 8)
max_batch_size = st.slider("Max batch size", 1, 64382, step=64)

memory_bw = st.slider("Accelerator HBM BW (GB/s)", 1, 4096, step=100, value=820)
flops = st.slider("Accelerator FLOPS (TFLOPS)", 1, 1024, step=16, value=197)

memory_bw = memory_bw * 1e9  # GB/s to B/s
flops = flops * 1e12  # TFLOPS to FLOPS

st.write("FLOPs/byte:", flops / memory_bw)



def roofline_df(flops, memory_bw):
    # Create a DataFrame with the roofline model
    arith_intensity = np.logspace(-1, 5, num=100)
    flops = np.minimum(flops, memory_bw * arith_intensity)
    return pd.DataFrame({
        'Arithmetic Intensity': arith_intensity,
        'FLOPs': flops,
    })    

def plot_roofline(df):
    # Create the roofline plot
    base = alt.Chart(df).encode(x=alt.X('Arithmetic Intensity:Q').scale(type="log"), y=alt.Y('FLOPs:Q').scale(type="log"))
    # Set a sensible baseline for the area that works with log scale
    return base.mark_line()
# Create the roofline DataFrame
df = roofline_df(flops, memory_bw)
# Plot the roofline
roofline_plot = plot_roofline(df)
st.altair_chart(roofline_plot, use_container_width=True)

@st.cache_data
def get_num_used_experts(num_experts, K, max_batch_size):
    batch_sizes = np.arange(1, max_batch_size + 1, 64)
    # assume uniform router distribution
    # calculate the number of used experts
    used_experts = []
    for b_size in batch_sizes:
        samps = []
        for _ in range(100):
            np.random.choice(num_experts, size=K, replace=False)

    pass




