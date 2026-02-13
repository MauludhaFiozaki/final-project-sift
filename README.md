# Final Project - SIFT Feature Detection, Matching, and Stitching

## Deskripsi Singkat
Project ini mengimplementasikan algoritma **SIFT (Scale-Invariant Feature Transform)** untuk:

1) Deteksi **keypoints** dan ekstraksi **descriptor**  
2) Feature matching antar dua citra (BFMatcher + Lowe Ratio Test)  
3) Estimasi homography menggunakan **RANSAC**  
4) Membuat panorama sederhana (image stitching)

Hasil output yang disimpan:
- outputs/keypoints_1.png
- outputs/keypoints_2.png
- outputs/matches.png
- outputs/panorama.png

---

## Struktur Folder

```
VISKOM/
├── images/
│   ├── 1.jpeg
│   └── 2.jpeg
├── outputs/
│   ├── keypoints_1.png
│   ├── keypoints_2.png
│   ├── matches.png
│   └── panorama.png
├── UAS.py
├── requirements.txt
└── README.md
```


## Cara Menjalankan (Replikasi)

### 1) Clone Repository

```bash
git clone https://github.com/mauludhafiozaki/final-project-sift.git
cd final-project-sift
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
```

### 3) Jalan kan Program

``` bash
python UAS.py


```md
---

## Output Program

### Keypoints Detection
![Keypoints1](outputs/keypoints_1.png)

### Feature Matching
![Matches](outputs/matches.png)

### Panorama Result
![Panorama](outputs/panorama.png)
