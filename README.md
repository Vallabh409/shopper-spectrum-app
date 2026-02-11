# Shopper Spectrum App - Customer Segmentation & Analysis

An advanced machine learning application for customer segmentation and behavioral analysis using clustering techniques. This application helps retail businesses understand customer patterns and segment them for targeted marketing strategies.

## 🎯 Features

- **Customer Segmentation**: Automatically segment customers into meaningful groups using K-Means clustering
- **RFM Analysis**: Recency, Frequency, Monetary value analysis for customer lifetime prediction
- **Behavioral Clustering**: Identify customer purchase patterns and preferences
- **Interactive Dashboard**: Visualize customer segments and metrics using Streamlit
- **Predictive Modeling**: Machine learning models for customer behavior prediction
- **Data Export**: Export segmentation results for further analysis

## 🛠️ Technology Stack

- **Language**: Python
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Deployment**: Streamlit Cloud / Render
- **Notebooks**: Jupyter Notebook

## 📋 Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Virtual Environment (recommended)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Vallabh409/shopper-spectrum-app.git
cd shopper-spectrum-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:
```bash
streaml it run app.py
```

5. Open your browser at `http://localhost:8501`

## 📁 Project Structure

```
shopper-spectrum-app/
├── Shopper_Spectrum.ipynb    # Data analysis & model development
├── app.py                    # Streamlit application
├── kmeans_model.pkl         # Trained K-Means model
├── scaler.pkl              # Feature scaler
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 💡 Usage

1. **Data Preparation**: Upload customer data (CSV format)
2. **Segment Analysis**: View automatically generated customer segments
3. **Cluster Visualization**: Interactive visualizations of customer clusters
4. **Metrics Dashboard**: Key metrics and insights for each segment
5. **Export Results**: Download segmentation results

## 👂 Algorithm Details

### K-Means Clustering
The application uses K-Means clustering to segment customers into k clusters based on their purchasing behavior. The optimal number of clusters is determined using:
- Elbow Method
- Silhouette Analysis

### Features Used
- Purchase frequency
- Average transaction value
- Customer lifetime value (CLV)
- Purchase recency
- Product category preferences

## 🔍 Results & Insights

The model identifies distinct customer segments, enabling:
- **Targeted Marketing**: Personalized campaigns for each segment
- **Customer Retention**: Focus on high-value customers
- **Product Recommendations**: Segment-specific recommendations
- **Churn Prediction**: Identify at-risk customers

## 🚀 Deployment

### Deploy on Streamlit Cloud:
```bash
streaml it deploy
```

### Deploy on Render:
1. Push code to GitHub
2. Connect Render to GitHub repository
3. Set command: `streamlit run app.py`

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Vallabh Bashyakarla**
- GitHub: [@Vallabh409](https://github.com/Vallabh409)
- LinkedIn: [Vallabh Bashyakarla](https://www.linkedin.com/in/vallabh-bashyakarla)

## 📧 Contact & Support

For questions, issues, or suggestions:
- Create an issue on GitHub
- Email: [your-email]

---

**Built with ❤️ for data-driven business insights**
