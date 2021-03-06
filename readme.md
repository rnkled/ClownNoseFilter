## Resumo

O Código representado a seguir, utiliza das Bibliotecas OpenCV, Pandas, Math e DLib para gerar uma representação de um nariz de Palhaço, e, aliado ao reconhecimento facial, inserir no rosto de uma pessoa em tempo real que aparecer na camera.

Dlib é uma biblioteca de software multiplataforma de uso geral escrita na linguagem de programação C++. Ela reune diversas ferramentas ligadas a inteligência artificial e Machine Learning para facilitar a resolução de problemas. 

Utilizamos ela em conjunto com o Python e a Biblioteca OpenCV para aplicar um reconhecimento facial a imagem capturada pela nossa webcam, e reconhecer a área do Nariz da face apresentada, através da tecnica de reconhecimento com Landmarks, que é um algoritmo que define certos pontos de Interesse no rosto identificado. Utilizando-se desta tecnologia, unimos a manipulação de vetores facilitada pelo módulo Pandas e math, para calcular o redimensionamento, e sobreposição das imagens que gostariamos de utilizar. Desta forma, conseguimos aplicar a imagem do "Nariz de Palhaço" com um vetor, sobre a imagem capturada pela webcam, calculando sua altura e largura, com base no posicionamento do nariz da face reconhecida, e do seu tamanho, tornando o vetor adaptável conforme a movimentação da face no video. 

![CodeImage](https://github.com/rnkled/ClownNoseFilter/blob/main/code.png)
