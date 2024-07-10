
import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import load_model, predict_model

# Cache the model loading to improve performance
@st.cache_data
def load_model_cached():
    return load_model('LGBM Model 08072024')


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to convert df to excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Main app function
def main():
    # Initial page configuration
    st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

    #model
    model = load_model_cached()

    # Title
    st.write("## ML model with pycaret")
    st.write("This model will use LGBM (Light Gradient Boosting Machine) to predict whether customers in the provided file will default on their payments or not.")
    st.markdown("---")
    
    # Sidebar for input method selection
    st.sidebar.write("## Input Method")
    input_method = st.sidebar.radio("Choose the input method:", ('Upload file', 'Manual input'))

    if input_method == 'Upload file':
        # File upload section
        st.sidebar.write("### Upload file")
        data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

        # Check if content is loaded into the application
        if data_file_1 is not None:
            try:
                df_credit = pd.read_feather(data_file_1)
                df_credit = df_credit.sample(50000)

                st.write("### Uploaded Data Preview")
                st.dataframe(df_credit.head())

                predict = predict_model(model, data=df_credit)

                st.write("### Prediction Results Preview")
                st.dataframe(predict.head())

                df_xlsx = to_excel(predict)
                st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        # Manual input section
        st.sidebar.write("### Manual Input")
        with st.sidebar.form(key='manual_input_form'):
            # Assuming the model requires specific inputs
            idade = st.number_input('idade', min_value=18, max_value=100, value=30)
            renda = st.number_input('renda', min_value=0, value=50000)
            sexo = st.selectbox('sexo', ['F', 'M'])
            posse_de_veiculo = st.selectbox('posse_de_veiculo', ['N', 'S'])
            posse_de_imovel = st.selectbox('posse_de_imovel', ['N', 'S'])
            qtd_filhos = st.number_input('qtd_filhos', min_value=0, max_value=10, value=0)
            tipo_renda = st.selectbox('tipo_renda', ['Assalariado', 'Empresário', 'Pensionista', 'Servidor público', 'Bolsista'])
            educacao = st.selectbox('educacao', ['Médio', 'Superior completo', 'Superior incompleto', 'Fundamental', 'Pós graduação'])
            estado_civil = st.selectbox('estado_civil', ['Casado', 'Solteiro', 'União', 'Separado', 'Viúvo'])
            tipo_residencia = st.selectbox('tipo_residencia', ['Casa', 'Com os pais', 'Governamental', 'Aluguel', 'Estúdio', 'Comunitário'])
            tempo_emprego = st.number_input('tempo_emprego', min_value=0, max_value=45, value=8)
            qt_pessoas_residencia = st.number_input('qt_pessoas_residencia', min_value=0, max_value=10, value=2)

            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            input_data = pd.DataFrame({
                'idade': [idade],
                'renda': [renda],
                'sexo': [sexo],
                'posse_de_veiculo': [posse_de_veiculo],
                'posse_de_imovel': [posse_de_imovel],
                'qtd_filhos': [qtd_filhos],
                'tipo_renda': [tipo_renda],
                'educacao': [educacao],
                'estado_civil': [estado_civil],
                'tipo_residencia': [tipo_residencia],
                'tempo_emprego': [tempo_emprego],
                'qt_pessoas_residencia': [qt_pessoas_residencia]
            })

            try:
                prediction = predict_model(model, data=input_data)
                st.write("### Manual Input Prediction Result")
                st.write(prediction)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()









