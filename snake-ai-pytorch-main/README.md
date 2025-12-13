Proyek ini mengimplementasikan game klasik Snake menggunakan framework web Flask (Python) dan melengkapinya dengan agen kecerdasan buatan (AI) yang dilatih menggunakan algoritma Deep Q-Learning (DQN). Aplikasi ini dirancang untuk menunjukkan kapabilitas AI dalam mengambil keputusan strategis untuk mencapai skor tertinggi dalam permainan.

Untuk menjalankan aplikasi ini secara lokal, Anda memerlukan:

Python 3 (disarankan 3.10+)

Git (Untuk mengkloning repositori)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di lingkungan lokal Anda (melalui localhost).

Langkah 1: Kloning Repositori
Buka Terminal (Linux/macOS) atau Command Prompt/PowerShell (Windows) dan jalankan perintah berikut:
git clone https://github.com/Yoruu-x1/snake_game_ai.git
cd snake_game_ai

Langkah 2: Buat dan Aktifkan Virtual Environment (Wajib)
Penggunaan Virtual Environment sangat disarankan untuk mengisolasi dependency proyek ini agar tidak mengganggu instalasi Python sistem Anda.

Langkah 3: Instal Dependensi
Dengan Virtual Environment aktif, instal semua pustaka yang diperlukan (termasuk PyTorch dan Flask) dari file requirements.txt:

Langkah 4: Jalankan Server Flask
Setelah semua dependency berhasil diinstal, jalankan server aplikasi:
python app.py

Langkah 5: Akses Aplikasi
Aplikasi Snake AI sekarang berjalan. Buka browser Anda dan navigasi ke URL berikut:

🔗 http://127.0.0.1:5000/

Anda akan melihat interface game dan agen AI Anda mulai bermain secara otomatis.
