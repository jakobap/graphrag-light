from setuptools import setup, find_packages

setup(
    name='graphrag_light',  # Choose a name for your library
    version='0.1', 
    packages=find_packages(),
    install_requires=[
        'google-cloud-aiplatform==1.59.0',
        'google-cloud-documentai==2.29.2',
        'google-auth==2.32.0',
        'google-api-python-client==2.129.0',
        'python-dotenv==1.0.1',
        'networkx==3.3',
        'matplotlib==3.9.1',
        'langfuse==2.39.2',
        'graspologic==3.4.1',
        'google-cloud-pubsub==2.19.6',
        'PyPDF2==3.0.1'
    ]
)