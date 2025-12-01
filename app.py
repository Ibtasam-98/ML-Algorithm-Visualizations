import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Set page configuration
st.set_page_config(
    page_title="Interactive ML 3D Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)


ALGORITHMS = {
    "Regression": {
        "Linear Regression": "linear",
    },
    "Classification": {
        "Logistic Regression": "logistic",
        "Decision Tree Classifier": "dtc",
        "Support Vector Machine (SVM)": "svm",
    }
}


# --- DATA GENERATION FUNCTIONS ---

def generate_linear_data(n_samples, noise_lin):
    """Generates synthetic 3D data for Linear Regression."""
    np.random.seed(42)
    X = np.random.uniform(-50, 50, (n_samples, 2))
    y = (X[:, 0] - 0.5 * X[:, 1] + 10 +
         np.random.normal(0, noise_lin, n_samples))
    return X, y


def generate_logistic_data(n_samples, boundary_skew):
    """Generates synthetic 3D data for Logistic Regression."""
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (n_samples, 2))
    linear_combination = (X[:, 0] + X[:, 1] * boundary_skew)
    probs = 1 / (1 + np.exp(-1.0 * linear_combination))
    y = (probs > 0.5 + np.random.uniform(-0.1, 0.1, n_samples)).astype(int)
    return X, y


def generate_circular_data(n_samples, noise_factor):
    """Generates synthetic 3D data with a non-linear (circular) boundary."""
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (n_samples, 2))
    center_distance = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (center_distance > 5).astype(int)
    y = np.logical_xor(y, (np.random.rand(n_samples) < noise_factor)).astype(int)
    return X, y


def generate_linear_separable_data(n_samples, noise_factor):
    """Generates linearly separable data for linear SVM."""
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (n_samples, 2))
    # Linear separation: y = 1 if x1 + x2 > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Add some noise
    noise_mask = np.random.rand(n_samples) < noise_factor
    y[noise_mask] = 1 - y[noise_mask]
    return X, y


def generate_xor_data(n_samples, noise_factor):
    """Generates XOR-like data for polynomial SVM."""
    np.random.seed(42)
    X = np.random.uniform(-10, 10, (n_samples, 2))
    # XOR pattern: class 1 when both coordinates have same sign
    y = (np.sign(X[:, 0]) == np.sign(X[:, 1])).astype(int)
    # Add some noise
    noise_mask = np.random.rand(n_samples) < noise_factor
    y[noise_mask] = 1 - y[noise_mask]
    return X, y


# --- PLOTTING UTILITY ---

def create_base_3d_plot(X, y, z_title, plot_title, is_binary=False, height=600):
    """Creates a base Plotly figure with data points."""
    colorscale = [[0, 'red'], [1, 'blue']] if is_binary else 'Viridis'

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=y,
        mode='markers',
        marker=dict(size=5, color=y, colorscale=colorscale, opacity=0.8),
        name='Data Points'
    ))

    fig.update_layout(
        title=f'<b>{plot_title}</b>',
        scene=dict(
            xaxis_title='Feature X1',
            yaxis_title='Feature X2',
            zaxis_title=z_title,
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=height
    )
    return fig


# --- RENDERING PAGES (ALGORITHMS) ---

def render_linear_regression(n_samples, lin_noise):
    """Renders the Linear Regression visualization page."""
    X, y = generate_linear_data(n_samples, lin_noise)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Create meshgrid for prediction surface
    x1_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 30)
    x2_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 30)
    X1_surf, X2_surf = np.meshgrid(x1_range, x2_range)
    X_surf_scaled = np.hstack([X1_surf.reshape(-1, 1), X2_surf.reshape(-1, 1)])
    Y_surf = model.predict(X_surf_scaled).reshape(X1_surf.shape)

    # Inverse transform the scaled features for original axis labels
    X1_surf_unscaled, X2_surf_unscaled = scaler.inverse_transform(X_surf_scaled).T.reshape(2, *X1_surf.shape)

    fig = create_base_3d_plot(X, y, 'Target Y (Continuous)', 'Linear Regression: Flat Plane Fit', is_binary=False)

    # Add Regression Surface
    fig.add_trace(go.Surface(
        x=X1_surf_unscaled, y=X2_surf_unscaled, z=Y_surf,
        colorscale='Viridis',
        opacity=0.6,
        showscale=False,
        name='Regression Plane'
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ### Concept
        Linear Regression fits a **flat, linear surface (a plane)** to continuous data points, minimizing the sum of squared errors (the vertical distance from each point to the plane). 
        This model assumes the relationship between features and the target is linear.
    """)


def render_logistic_regression(n_samples, boundary_skew):
    """Renders the Logistic Regression visualization page."""
    X, y = generate_logistic_data(n_samples, boundary_skew)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_scaled, y)

    # Create meshgrid for prediction surface
    x1_range = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 30)
    x2_range = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 30)
    X1_surf, X2_surf = np.meshgrid(x1_range, x2_range)
    X_surf_scaled = np.hstack([X1_surf.reshape(-1, 1), X2_surf.reshape(-1, 1)])

    # Probability Surface P(Y=1)
    Z_prob = model.predict_proba(X_surf_scaled)[:, 1].reshape(X1_surf.shape)

    # Inverse transform for original axis labels
    X1_surf_unscaled, X2_surf_unscaled = scaler.inverse_transform(X_surf_scaled).T.reshape(2, *X1_surf.shape)

    fig = create_base_3d_plot(X, y, 'P(Y=1) / Target Z (0 or 1)', 'Logistic Regression: Sigmoid Probability Surface',
                              is_binary=True)

    # Add Probability Surface
    fig.add_trace(go.Surface(
        x=X1_surf_unscaled, y=X2_surf_unscaled, z=Z_prob,
        colorscale='RdBu',
        opacity=0.7,
        showscale=False,
        name='Probability Surface P(Y=1)',
    ))

    # Add Decision Boundary Plane (Z=0.5)
    fig.add_trace(go.Surface(
        x=X1_surf_unscaled, y=X2_surf_unscaled, z=0.5 * np.ones(Z_prob.shape),
        colorscale=[[0, 'gray'], [1, 'gray']],
        opacity=0.2,
        showscale=False,
        name='P=0.5 Boundary',
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ### Concept
        Logistic Regression fits an **S-shaped probability surface (sigmoid)**, bounded between 0 and 1. It models the probability of a point belonging to class 1. The faint grey plane at $Z=0.5$ is the **linear decision boundary** where the model transitions from predicting class 0 to class 1.
    """)


def render_decision_tree(n_samples, dtc_max_depth, dtc_noise_factor):
    """Renders the Decision Tree Classifier visualization page."""
    X, y = generate_circular_data(n_samples, dtc_noise_factor)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the Decision Tree
    model = DecisionTreeClassifier(max_depth=dtc_max_depth, random_state=42)
    model.fit(X_scaled, y)

    # Create meshgrid for prediction surface
    x1_range = np.linspace(X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1, 100)
    x2_range = np.linspace(X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1, 100)
    X1_surf, X2_surf = np.meshgrid(x1_range, x2_range)
    X_surf_scaled = np.hstack([X1_surf.reshape(-1, 1), X2_surf.reshape(-1, 1)])

    # Prediction surface (Z is 0 or 1)
    Z_pred = model.predict(X_surf_scaled).reshape(X1_surf.shape)

    # Inverse transform for original axis labels
    X1_surf_unscaled, X2_surf_unscaled = scaler.inverse_transform(X_surf_scaled).T.reshape(2, *X1_surf.shape)

    fig = create_base_3d_plot(X, y, 'Predicted Class (0 or 1)', f'Decision Tree: Max Depth = {dtc_max_depth}',
                              is_binary=True)

    # Add Classification Surface (Stepped Surface)
    fig.add_trace(go.Surface(
        x=X1_surf_unscaled, y=X2_surf_unscaled, z=Z_pred,
        colorscale=[[0, 'rgba(255,100,100,0.5)'], [1, 'rgba(100,100,255,0.5)']],
        opacity=0.7,
        showscale=False,
        name='Decision Surface',
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
        ### Concept
        The Decision Tree recursively splits the feature space into **rectangular regions** (visible as distinct steps in the 3D surface) to classify data. 

        The **Max Depth** parameter ($D={dtc_max_depth}$) controls the complexity: 
        * **Low Depth** creates simple, coarse boundaries (potential underfitting).
        * **High Depth** creates highly complex, jagged boundaries, often overfitting to the noise in the data.
    """)


def render_svm(n_samples, svm_kernel, svm_c, svm_gamma, svm_degree, noise_factor):
    """Renders the Support Vector Machine (SVM) visualization page."""

    # Generate appropriate data for each kernel type
    if svm_kernel == 'linear':
        X, y = generate_linear_separable_data(n_samples, noise_factor)
        data_description = "Linearly Separable Data"
    elif svm_kernel == 'poly':
        X, y = generate_xor_data(n_samples, noise_factor)
        data_description = "XOR-like Data (Requires Polynomial Separation)"
    else:  # rbf
        X, y = generate_circular_data(n_samples, noise_factor)
        data_description = "Circular Data (Requires Non-linear Separation)"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Kernel setup
    if svm_kernel == 'rbf':
        model = SVC(kernel=svm_kernel, C=svm_c, gamma=svm_gamma, random_state=42)
    elif svm_kernel == 'poly':
        model = SVC(kernel=svm_kernel, C=svm_c, degree=svm_degree, random_state=42)
    else:  # linear
        model = SVC(kernel=svm_kernel, C=svm_c, random_state=42)

    model.fit(X_scaled, y)

    # Create meshgrid for prediction surface
    x1_range = np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 100)
    x2_range = np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 100)
    X1_surf, X2_surf = np.meshgrid(x1_range, x2_range)
    X_surf_scaled = np.hstack([X1_surf.reshape(-1, 1), X2_surf.reshape(-1, 1)])

    # Get decision function values for better visualization
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(X_surf_scaled).reshape(X1_surf.shape)
        # Convert to probability-like values for better visualization
        Z_vis = 1 / (1 + np.exp(-Z))  # Sigmoid transformation for better color scaling
    else:
        Z_vis = model.predict(X_surf_scaled).reshape(X1_surf.shape)

    # Inverse transform for original axis labels
    X1_surf_unscaled, X2_surf_unscaled = scaler.inverse_transform(X_surf_scaled).T.reshape(2, *X1_surf.shape)

    # Create the base plot
    fig = create_base_3d_plot(X, y, 'Decision Function / Predicted Class',
                              f'SVM: {svm_kernel.upper()} Kernel - {data_description}',
                              is_binary=True)

    # Add Decision Surface with continuous colors based on decision function
    if hasattr(model, 'decision_function'):
        # Use continuous colors for decision function
        fig.add_trace(go.Surface(
            x=X1_surf_unscaled, y=X2_surf_unscaled, z=Z_vis,
            colorscale='RdBu',
            opacity=0.7,
            showscale=True,
            name='Decision Function',
            colorbar=dict(title="Decision<br>Function")
        ))

        # Add support vectors
        support_vectors = model.support_vectors_
        support_vectors_unscaled = scaler.inverse_transform(support_vectors)
        y_support = y[model.support_]

        fig.add_trace(go.Scatter3d(
            x=support_vectors_unscaled[:, 0],
            y=support_vectors_unscaled[:, 1],
            z=np.zeros(len(support_vectors_unscaled)),  # Place at bottom
            mode='markers',
            marker=dict(
                size=8,
                color=y_support,
                colorscale=[[0, 'darkred'], [1, 'darkblue']],
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            name='Support Vectors'
        ))
    else:
        # Fallback to discrete colors
        fig.add_trace(go.Surface(
            x=X1_surf_unscaled, y=X2_surf_unscaled, z=Z_vis,
            colorscale=[[0, 'rgba(255,100,100,0.5)'], [1, 'rgba(100,100,255,0.5)']],
            opacity=0.7,
            showscale=False,
            name='Decision Surface',
        ))

    st.plotly_chart(fig, use_container_width=True)

    # Display kernel-specific information
    kernel_info = {
        'linear': "**Linear Kernel**: Finds the optimal flat plane that maximizes the margin between classes. Best for linearly separable data.",
        'poly': f"**Polynomial Kernel (Degree {svm_degree})**: Uses polynomial functions to create curved boundaries. Can capture more complex patterns than linear kernels.",
        'rbf': f"**RBF Kernel (Gamma {svm_gamma})**: Creates highly flexible, complex boundaries using radial basis functions. Excellent for highly non-linear data like the circular pattern shown."
    }

    st.markdown(f"""
        ### Concept
        Support Vector Machine (SVM) finds the optimal hyperplane that maximizes the margin between classes.

        {kernel_info[svm_kernel]}

        **C Parameter (C={svm_c})**: Controls the trade-off between achieving a wide margin and minimizing classification errors:
        * **Low C**: Wider margin, more tolerant of misclassifications (regularized)
        * **High C**: Narrower margin, stricter about misclassifications (may overfit)

        **Support Vectors** (diamond markers) are the data points that define the margin boundary. The decision surface shape depends entirely on these critical points.
    """)


# --- MAIN APPLICATION LOGIC ---

def main():
    st.title("Interactive ML Algorithm Visualizer (3D)")
    st.markdown("Use the sidebar to select an algorithm and adjust its parameters.")

    # 1. Sidebar Navigation
    with st.sidebar:
        st.header("Algorithm Selection")

        # Select Algorithm Group (e.g., Regression, Classification)
        algo_group = st.selectbox(
            "1. Select Algorithm Group",
            list(ALGORITHMS.keys()),
            key='algo_group'
        )

        # Select Specific Algorithm
        algo_name = st.selectbox(
            "2. Select Algorithm",
            list(ALGORITHMS[algo_group].keys()),
            key='algo_name'
        )

        # Store the simple identifier for the rendering function
        algo_key = ALGORITHMS[algo_group][algo_name]

        st.header(f"Parameters for {algo_name}")

        # General Data Parameter
        n_samples = st.slider("Number of Data Points", 100, 500, 200, 50, key='n_samples')

        # --- Algorithm Specific Parameters ---
        params = {'n_samples': n_samples}

        if algo_key == "linear":
            lin_noise = st.slider("Noise Level (Linear Data)", 5, 40, 15, 5,
                                  help="Controls scatter around the regression plane.")
            params['lin_noise'] = lin_noise

        elif algo_key == "logistic":
            boundary_skew = st.slider("Boundary Skew (X2 Weight)", -2.0, 2.0, 1.0, 0.1,
                                      help="Adjusts the angle/tilt of the decision boundary.")
            params['boundary_skew'] = boundary_skew

        elif algo_key == "dtc":
            dtc_max_depth = st.slider("Max Depth", 1, 15, 5, 1,
                                      help="Controls the complexity and potential for overfitting.")
            dtc_noise_factor = st.slider("Data Noise Factor", 0.05, 0.4, 0.1, 0.05, key='dtc_noise_factor',
                                         help="Controls overlap between the classes in the generated data.")
            params['dtc_max_depth'] = dtc_max_depth
            params['dtc_noise_factor'] = dtc_noise_factor

        elif algo_key == "svm":
            svm_kernel = st.selectbox("Kernel Type", ['rbf', 'linear', 'poly'], key='svm_kernel')
            svm_c = st.select_slider("C (Regularization)", options=[0.01, 0.1, 1, 10, 100], value=1.0,
                                     help="Penalty parameter for error term.")
            noise_factor = st.slider("Data Noise Factor", 0.05, 0.4, 0.1, 0.05, key='svm_noise_factor',
                                     help="Controls overlap between the classes in the generated data.")

            # Initialize optional parameters
            svm_gamma = 'scale'
            svm_degree = 3

            if svm_kernel == 'rbf':
                svm_gamma = st.select_slider("Gamma (RBF Kernel)", options=[0.01, 0.1, 1, 10, 100], value=0.1,
                                             help="Kernel coefficient for RBF.")

            elif svm_kernel == 'poly':
                svm_degree = st.slider("Degree (Poly Kernel)", 2, 5, 3, 1, help="Degree of the polynomial function.")

            params['svm_kernel'] = svm_kernel
            params['svm_c'] = svm_c
            params['svm_gamma'] = svm_gamma
            params['svm_degree'] = svm_degree
            params['noise_factor'] = noise_factor

        st.markdown("---")
        st.markdown("Refresh the page to reset parameters.")

    # 2. Main Content Rendering
    st.subheader(f"Visualization: {algo_name}")

    if algo_key == "linear":
        render_linear_regression(**params)
    elif algo_key == "logistic":
        render_logistic_regression(**params)
    elif algo_key == "dtc":
        render_decision_tree(**params)
    elif algo_key == "svm":
        render_svm(**params)
    else:
        st.error("Please select a valid algorithm from the sidebar.")


if __name__ == "__main__":
    main()