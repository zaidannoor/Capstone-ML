# Menggunakan Python 3.9.2 sebagai base image
FROM python:3.9.2

# Set working directory di dalam container
WORKDIR /app

# Menyalin seluruh kode aplikasi ke dalam container
COPY . ./

# COPY requirements.txt .

# Menginstal dependensi Python menggunakan pip
RUN pip install tensorflow==2.12.0 --default-timeout=100 future 
RUN pip install tensorflow-intel
RUN pip install --no-cache-dir -r requirements.txt



# Menjalankan aplikasi Flask ketika container dijalankan

CMD ["python", "main.py"]
EXPOSE 6060