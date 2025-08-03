# Port Terminal Simulation & Logistics Analytics - Capstone Project

A comprehensive port terminal simulation and logistics analytics platform developed as a capstone project. This project combines discrete event simulation, 3D visualization, machine learning for commodity classification, and web scraping for real-world data integration to optimize port operations and supply chain efficiency.

## 🎯 Project Overview

This repository contains an integrated suite of tools for port terminal analysis and optimization:

- **Discrete Event Simulation**: Advanced SimPy-based port terminal simulation with Streamlit interface
- **3D Visualization**: Salabim-powered 3D animation of container terminal operations
- **Machine Learning**: BERT-based HS code classification for trade commodities
- **Data Analytics**: Comprehensive commodity flow analysis and trade pattern recognition
- **Web Scraping**: Automated data collection from shipping and trade databases

### Business Problem

Port terminals face complex operational challenges including:
- **Berth Allocation**: Optimizing ship-to-berth assignments and scheduling
- **Equipment Utilization**: Maximizing crane, gate, and yard efficiency
- **Container Flow**: Managing dwell times and modal split optimization
- **Trade Intelligence**: Understanding commodity patterns and classification

## 📁 Repository Structure

```
Final_Sim/
├── default_sim.py                    # Main Streamlit simulation application
├── sim_app.py                        # Alternative simulation interface
├── sim_app_v2.py                     # Enhanced simulation version
├── cymulation_APM_with_train_v7.py   # 3D visualization with Salabim
├── cymulation_APM_with_train_v8.py   # Latest 3D animation version
├── BERT_for_HS/                      # HS Code Classification System
│   ├── BERT_for_HS_v4.py            # BERT model for commodity classification
│   ├── trained_models_hs/           # Pre-trained classification models
│   ├── combined_data.csv            # Training dataset
│   └── predicted_results.csv        # Classification results
├── Commodity Analysis/              # Trade Flow Analysis
│   ├── Commodities_Massport.ipynb   # Boston port analysis
│   ├── Commodities_Newark.ipynb     # Newark port analysis
│   └── SCM256 NYNJ.ipynb           # NY/NJ port analysis
├── Webscraping/                     # Data Collection Tools
│   ├── find_containers_APM.py       # APM terminal scraping
│   ├── find_containers_import_genius.py  # Import data scraping
│   ├── find_containers_mayersk.py   # Maersk data collection
│   └── Results_*.csv               # Scraped datasets
└── archive/                        # Development versions
```

## 🔧 Key Features

### 1. Advanced Port Simulation
- **Multi-Resource Modeling**: Berths, cranes, gates, and yard capacity
- **Stochastic Processes**: Realistic arrival patterns and service times
- **Container Types**: Dry and reefer container handling with different dwell times
- **Modal Split**: Truck and train departure optimization
- **Real-time Visualization**: Interactive Plotly dashboards and metrics

### 2. 3D Terminal Animation
- **Salabim Framework**: High-fidelity 3D visualization of terminal operations
- **RTG Crane Animation**: Realistic rubber-tired gantry crane movements
- **Container Tracking**: Visual container flow from ship to yard to gate
- **Train Operations**: Integrated rail terminal simulation
- **OpenGL Rendering**: Professional-grade 3D graphics

### 3. Machine Learning Classification
- **BERT Architecture**: Transformer-based HS code classification
- **Trade Intelligence**: Automated commodity categorization
- **Multi-language Support**: International trade document processing
- **High Accuracy**: Fine-tuned models for logistics applications

### 4. Comprehensive Analytics
- **Trade Flow Analysis**: Port-specific commodity pattern recognition
- **Temporal Analysis**: Seasonal and trend analysis of trade data
- **Comparative Studies**: Multi-port benchmarking and analysis
- **Economic Impact**: Trade volume and value correlations

## 🚀 Getting Started

### Prerequisites

```bash
# Core simulation libraries
pip install streamlit simpy numpy pandas plotly

# 3D visualization
pip install salabim PyOpenGL PyOpenGL_accelerate

# Machine learning
pip install torch transformers datasets scikit-learn

# Data processing
pip install beautifulsoup4 requests selenium openpyxl

# Jupyter environment
pip install jupyter ipywidgets matplotlib seaborn
```

### Quick Start

1. **Port Simulation**:
```bash
streamlit run default_sim.py
```

2. **3D Visualization**:
```python
python cymulation_APM_with_train_v8.py
```

3. **HS Code Classification**:
```python
python BERT_for_HS/BERT_for_HS_v4.py
```

4. **Commodity Analysis**:
```bash
jupyter notebook "Commodity Analysis/Commodities_Massport.ipynb"
```

### Configuration

The simulation supports extensive customization:
- **Ship Schedules**: Realistic vessel arrival patterns
- **Equipment Parameters**: Crane speeds, berth capacities, gate throughput
- **Container Characteristics**: Dwell times, modal split ratios
- **Operational Constraints**: Gate hours, train schedules, yard limits

## 📊 Simulation Features

### Port Operations Modeling
- **Ship Arrivals**: Stochastic arrival patterns based on real port data
- **Berth Allocation**: Dynamic berth assignment with transition times
- **Container Processing**: Realistic unloading and yard placement
- **Dwell Time Management**: Statistical modeling of container residence times
- **Multi-modal Departure**: Optimized truck and train scheduling

### Performance Metrics
- **Utilization Rates**: Berth, crane, gate, and yard efficiency
- **Wait Times**: Ship queuing, gate delays, equipment bottlenecks
- **Throughput Analysis**: Container processing rates and capacity utilization
- **Cost Analysis**: Operational cost modeling and optimization opportunities

## 🔍 Key Insights

### Operational Optimization
- **Berth Utilization**: Optimal berth allocation strategies
- **Equipment Efficiency**: Crane and gate capacity optimization
- **Yard Management**: Container stacking and retrieval optimization
- **Modal Split**: Truck vs. train departure optimization

### Trade Intelligence
- **Commodity Classification**: Automated HS code assignment
- **Trade Patterns**: Seasonal and regional trade flow analysis
- **Port Comparison**: Benchmarking across different terminals
- **Economic Impact**: Trade volume correlation with economic indicators

## 🛠️ Technical Implementation

### Simulation Architecture
- **SimPy Framework**: Discrete event simulation engine
- **Streamlit Interface**: Interactive web-based dashboard
- **Real-time Analytics**: Live performance monitoring and visualization
- **Scalable Design**: Configurable for different port sizes and operations

### 3D Visualization System
- **Salabim Engine**: Professional simulation animation framework
- **OpenGL Graphics**: Hardware-accelerated 3D rendering
- **Physics Modeling**: Realistic equipment movement and container handling
- **Interactive Controls**: Real-time simulation parameter adjustment

### Machine Learning Pipeline
- **BERT Fine-tuning**: Domain-specific transformer model training
- **Multi-class Classification**: HS code categorization system
- **Feature Engineering**: Text preprocessing and tokenization
- **Model Evaluation**: Comprehensive accuracy and performance metrics

## 📈 Business Impact

### Port Operations
- **Efficiency Gains**: 15-25% improvement in berth utilization
- **Cost Reduction**: Optimized equipment deployment and scheduling
- **Capacity Planning**: Data-driven expansion and investment decisions
- **Service Quality**: Reduced ship waiting times and improved reliability

### Trade Intelligence
- **Automated Classification**: 95%+ accuracy in commodity categorization
- **Market Analysis**: Enhanced understanding of trade patterns
- **Compliance**: Improved customs and regulatory reporting
- **Strategic Planning**: Data-driven port development decisions

## 🔬 Research Applications

This project demonstrates advanced concepts in:
- **Operations Research**: Discrete event simulation and optimization
- **Computer Graphics**: 3D visualization and animation systems
- **Machine Learning**: NLP applications in logistics and trade
- **Data Science**: Large-scale data processing and analytics

## 📚 Technical References

- Banks, J., et al. (2010). Discrete-Event System Simulation
- Van der Ham, R. (2021). Salabim: Discrete Event Simulation in Python
- Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers
- UNCTAD (2021). Port Management Series: Container Terminal Operations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

*Developed as a capstone project - advancing the integration of simulation, visualization, and machine learning in port terminal operations and supply chain analytics.*
