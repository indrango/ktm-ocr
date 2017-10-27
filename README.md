## KTM OCR ##

Phase 1 
Mendapatkan kontur dari data npm, nama, jurusan, fakultas.
Dilakukan beberapa image processing untuk mendapatkan contour dari
data tersebut untuk membuat mesin dapat menentukan bagian mana
yang akan diambil.

Phase 2:
Setelah didapatkan contour, contour ini sudah berupa kotak npm,
nama, jurusan, fakultas yang dilakukan image processing untuk memperjelas
pengambilan data tersebut.

Setelah itu digunakan library pytesseract untuk mengekstrak data contour
tersebut menjadi text. 

Output diterminal :

NPM         : (NPM)
NAMA        : (NAMA)
JURUSAN     : (JURUSAN)
FAKULTAS    : (FAKULTAS)# KTM OCR
