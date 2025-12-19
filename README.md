 DATASET ТАНИЛЦУУЛГА

IMDB Large Movie Review Dataset нь sentiment analysis судалгаанд өргөн
хэрэглэгддэг, ( <https://ai.stanford.edu/~amaas/data/sentiment/> ) олон
улсын түвшинд хүлээн зөвшөөрөгдсөн өгөгдлийн багц юм. Энэхүү dataset-ийг
2011 онд Maas болон түүний хамтран ажиллагчид боловсруулж,
ai.stanford.edu/\~amaas/data/sentiment/ хаягаар олон нийтэд нээлттэй
болгосон. Анх уг өгөгдлийг бүтээхдээ судлаачид тухайн үеийн sentiment
analysis судалгаанд тулгарч байсан хамгийн том бэрхшээлүүд болох
өгөгдлийн хэмжээ бага, ангиллын тэнцвэр алдагдсан байдал, мөн туршилтын
үр дүнг давтан батлах боломж хязгаарлагдмал байсныг шийдэх зорилго
тавьсан байдаг. Иймээс энэхүү dataset нь цэвэр, тэнцвэртэй, дахин
ашиглахад тохиромжтой benchmark өгөгдөл болж чадсан.

Уг өгөгдлийн багц нь нийт 50,000 ширхэг тэмдэглэгээтэй (label-тэй)
киноны тойм болон нэмэлтээр 50,000 ширхэг тэмдэглэгээгүй тоймоос
бүрддэг. Тэмдэглэгээтэй хэсэг нь сургалтын болон тестийн өгөгдөлд яг
тэнцүү хуваагдсан бөгөөд тус бүр 25,000 тоймоос бүрдэнэ. Sentiment
ангиллыг хэрэглэгчийн өгсөн үнэлгээнд тулгуурлан тодорхойлсон ба 7 болон
түүнээс дээш үнэлгээтэй тоймыг эерэг, 4 болон түүнээс доош үнэлгээтэй
тоймыг сөрөг гэж ангилсан. Харин 5 болон 6 гэсэн дунд үнэлгээтэй
тоймуудыг зориуд хассан нь sentiment-ийн хоёрдмол, тодорхой бус байдлыг
арилгах зорилготой юм. Үүний үр дүнд энэхүү dataset нь маш тодорхой,
хоёр ангилалттай sentiment classification хийхэд тохиромжтой бүтэцтэй
болсон.

Текстийн агуулгын хувьд IMDB dataset нь хэрэглэгчийн өөрийн санаа бодлыг
чөлөөтэй илэрхийлсэн бодит бичвэрүүдээс бүрддэг. Нэг тойм дунджаар
200--300 орчим үгтэй бөгөөд зарим тохиолдолд илүү урт, дэлгэрэнгүй
тайлбар агуулсан байдаг. Хэл найруулга нь албан бус, ярианы хэллэг,
егөөдөл, үгүйсгэл, сэтгэл хөдлөл ихтэй байдгаараа онцлог. Энэ нь тухайн
өгөгдлийг зөвхөн үгийн давтамжид суурилсан аргаар төдийгүй өгүүлбэрийн
дараалал, контекстийг харгалзан үздэг илүү нарийн загваруудаар судлах
боломжийг олгодог. Ийм учраас энэхүү dataset нь уламжлалт Bag-of-Words
болон TF-IDF аргаас эхлээд LSTM, Transformer, BERT зэрэг орчин үеийн
deep learning загваруудад хүртэл өргөн ашиглагдсаар ирсэн.

IMDB Large Movie Review Dataset-ийг дэлхийн олон улс орны судлаачид, их
сургуулийн багш оюутнууд, мөн аж үйлдвэрийн салбарын мэргэжилтнүүд
ашигладаг. Академик орчинд уг өгөгдлийг ихэвчлэн sentiment analysis
алгоритмуудын гүйцэтгэлийг харьцуулах, шинэ загварын үр нөлөөг шалгах
benchmark болгон ашигладаг.

Практик хэрэглээний хувьд уг өгөгдөл нь зөвхөн киноны үнэлгээтэй
хязгаарлагдахгүй. Олон судалгаанд IMDB dataset-ийг ашиглан боловсруулсан
аргачлалуудыг бүтээгдэхүүний сэтгэгдэл, сошиал медиа пост, хэрэглэгчийн
санал хүсэлт зэрэг бусад домэйнд шилжүүлэн ашиглах боломжийг судалдаг.
Өөрөөр хэлбэл, киноны тойм дээр сурсан sentiment загварууд нь
хэрэглэгчийн ерөнхий сэтгэл хандлагыг ойлгох, зах зээлийн судалгаа хийх,
хэрэглэгчийн зан төлөвийг таамаглах зэрэг өргөн хүрээний зорилгод
ашиглагддаг. Иймээс IMDB Large Movie Review Dataset нь зөвхөн нэг
салбарын өгөгдөл биш, харин sentiment analysis судалгааны суурь,
туршилтын үндсэн орчин болж чадсан гэж үзэж болно.

(https://ai.stanford.edu/~amaas/data/sentimen2. DATASET АШИГЛАГДСАН БАЙДАЛ

2.1 [Semisupervised Autoencoder for Sentiment
Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/10159)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Zhengdong Zhai болон Bing
Liu Zhang**** нар ****2016 онд**** хийж, ****AAAI Conference on
Artificial Intelligence****-д нийтэлсэн. Судалгаа нь АНУ-ын судалгааны
байгууллагуудын хүрээнд хийгдсэн бөгөөд sentiment analysis-д зориулсан
representation learning-ийн асуудлыг шийдэхэд чиглэсэн. Уламжлалт
unsupervised autoencoder нь текст дэх бүх үгийг ижил түвшинд reconstruct
хийдэг тул sentiment ангилалд чухал үгсийг ялгаж чаддаггүй гэж зохиогчид
үзсэн. Үүнийг шийдэхийн тулд label-тэй өгөгдлөөс сурсан ангилагчийн
мэдээллийг autoencoder-ийн сургалтад ашиглах semisupervised аргыг санал
болгосон.

Embedding / representation learning арга: Document-level embedding
сурсан. Текстийг эхлээд Bag-of-Words (unigram, bigram) хэлбэрээр
төлөөлсөн. Үүний дараа autoencoder-ийн hidden layer-ийг document
embedding гэж үзсэн. Хоёр үндсэн embedding ойлголт ашигласан: уламжлалт
denoising autoencoder дээр суурилсан embedding болон санал болгосон
semisupervised Bregman divergence autoencoder дээр суурилсан embedding.
Word2Vec, GloVe зэрэг pretrained word embedding ашиглаагүй.

Hyperparameter-ууд: Denoising autoencoder-ийн hidden layer-ийн хэмжээ
2000, semisupervised autoencoder-ийн hidden layer-ийн хэмжээ 200 байсан.
Encoder талд ReLU, decoder талд sigmoid activation ашигласан. Optimizer
нь mini-batch stochastic gradient descent, momentum 0.9. Denoising
сургалтын хүрээнд оролтын өгөгдөлд санамсаргүй noise нэмсэн. Ангилагчийн
Bayesian хувилбарт temperature параметр β-г 10⁴--10⁸ хооронд туршсан.
Learning rate, batch size, epoch-ийн тоог paper-т нарийвчлан дурдаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
талд нэг hidden layer-тэй autoencoder ашигласан. Machine Learning талд
squared hinge loss-тэй шугаман Support Vector Machine ашиглаж sentiment
classifier сургасан. Bayesian өргөтгөлтэй хувилбарт Laplace
approximation ашиглан ангилагчийн жингүүдийн магадлалын тархалтыг
тооцсон.

Ашиглагдсан Computer Science аргууд: Feature engineering-д Bag-of-Words,
unigram болон bigram ашигласан. Text normalization-д логарифмд суурилсан
normalization хэрэглэсэн. Semisupervised learning, representation
learning, generative болон discriminative model-уудыг хослуулах hybrid
аргачлал ашигласан. Мөн statistical divergence ойлголтод суурилсан
Bregman divergence loss хэрэглэсэн.

Үр дүн, ашигласан үнэлгээний арга : Үр дүнг голчлон classification error
rate болон accuracy ашиглан үнэлсэн. F1-score, precision, recall-ийг
үндсэн metric болгон ашиглаагүй. IMDB Large Movie Review Dataset дээр
Bag-of-Words + SVM суурь загвар ойролцоогоор 92.6% accuracy (7.4% error)
үзүүлсэн. Уламжлалт unsupervised denoising autoencoder ашигласан үед
accuracy ойролцоогоор 92.5--92.7% орчим байсан. Харин санал болгосон
semisupervised Bregman divergence autoencoder нь ангиллын алдааг
6.2--6.5% хүртэл бууруулж, accuracy-ийг ойролцоогоор 93.5--93.8%
хүргэсэн. Тэмдэглэгээгүй өгөгдлийг сургалтад ашигласнаар accuracy
ойролцоогоор 0.5--0.8%-иар нэмэгдсэн. Amazon-ийн Books, DVD, Electronics
dataset-үүд дээр baseline-тай харьцуулахад 1--2%-ийн нэмэлт сайжруулалт
тогтвортой ажиглагдсан.

2.2 [Text based Sentiment Analysis using
LSTM](https://pdfs.semanticscholar.org/0027/d572e43d0c120d59e81c228f2a17b3b05006.pdf)

Товч танилцуулга : Энэхүү судалгааны ажлыг ****Murthy, Kumar, Rao****
нар ****2020 онд**** хийж, ****International Journal of Engineering
Research & Technology (IJERT)**** сэтгүүлд нийтэлсэн. Судалгаа нь
текстийн sentiment analysis-д уламжлалт машин сургалтын аргууд дараалсан
өгөгдлийн урт хугацааны хамаарлыг бүрэн барьж чаддаггүй асуудлыг
шийдэхэд чиглэсэн. Үүний тулд зохиогчид recurrent neural network-ийн
сайжруулсан хувилбар болох LSTM загварыг ашиглан кино болон
бүтээгдэхүүний review-үүдийн эерэг, сөрөг sentiment-ийг ангилах зорилт
тавьсан.

Embedding / representation learning арга: Судалгаанд word-level
embedding ашигласан. Текстийг эхлээд tokenization хийж integer sequence
болгон хувиргасан. Үүний дараа trainable embedding layer ашиглан үг
бүрийг dense vector хэлбэрээр төлөөлсөн. Embedding нь тухайн
загвартайгаа хамт сургагдсан бөгөөд Word2Vec, GloVe зэрэг pretrained
embedding ашиглаагүй. Дараагийн шатанд LSTM-ийн hidden state-ийг
өгөгдлийн sequence-д суурилсан document representation гэж үзсэн.

Hyperparameter-ууд: Vocabulary size нь ойролцоогоор 6000 үг байсан.
Embedding dimension 100 (зарим туршилтад 128 гэж дурдсан). LSTM
layer-ийн hidden unit-ийн тоо 128. Dropout 0.2 ашигласан. Batch size
500. Epoch-ийн тоог 1-ээс 10 хүртэл өөрчилж туршсан. Optimizer нь Adam,
loss function нь categorical cross-entropy байсан. Learning rate-ийн
утгыг paper-т нарийвчлан заагаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
талд Embedding layer, нэг LSTM layer, fully connected dense layer болон
softmax output layer бүхий нейрон сүлжээ ашигласан. Уламжлалт Machine
Learning алгоритм ашиглаагүй бөгөөд судалгаа нь цэвэр deep learning-д
суурилсан. LSTM нь vanishing gradient асуудлыг шийдэж, урт дараалсан
текстээс утгын хамаарлыг сурч чаддаг гэдгийг онцолсон.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
үндсэн аргууд болох tokenization, padding, sequence modeling ашигласан.
Sequential data processing, backpropagation through time, gradient-based
optimization зэрэг алгоритмын түвшний аргуудыг хэрэглэсэн. Мөн
supervised learning орчинд neural network training хийсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон accuracy ашиглан
үнэлсэн. Precision, recall, F1-score-ийг үндсэн metric болгон
ашиглаагүй. IMDB Movie Review Dataset дээр хийсэн туршилтаар 1 epoch
сургалтын дараа accuracy ойролцоогоор 56.1% байсан бол epoch-ийн тоо
нэмэгдэх тусам гүйцэтгэл эрс сайжирсан. 5 epoch-ийн дараа accuracy 96.1%
хүрч, 10 epoch-ийн дараа ойролцоогоор 99.5% accuracy үзүүлсэн гэж
тайлагнасан. Судалгаанд LSTM нь сургалтын өгөгдөл хангалттай үед
sentiment classification-д маш өндөр гүйцэтгэл үзүүлж чаддагийг
харуулсан.

2.3 [Sentiment Classification of Movie Review using Machine Learning
Approach](https://www.researchgate.net/profile/Ashish-Rajaram/publication/385707147_Sentiment_Classification_of_Movie_Review_using_Machine_Learning_Approach/links/6731e453ecbbde716b6834b1/Sentiment-Classification-of-Movie-Review-using-Machine-Learning-Approach.pdf)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Lahase болон Deshmukh****
нар ****2022 онд**** хийж, ****International Research Journal of
Engineering and Technology (IRJET)****-д нийтэлсэн. Судалгааны зорилго
нь киноны review-үүдийн sentiment-ийг уламжлалт машин сургалтын аргууд
ашиглан ангилах бөгөөд текстийн урьдчилсан боловсруулалт, feature
extraction болон dimensionality reduction-ийн нөлөөг судлахад чиглэсэн.
Зохиогчид deep learning ашиглахгүйгээр, зөв тохируулсан classical
machine learning загварууд sentiment analysis-д ямар гүйцэтгэл үзүүлж
болохыг харуулахыг зорьсон.

Embedding / representation learning арга: Судалгаанд neural embedding
ашиглаагүй. Текстийг Bag-of-Words загварт тулгуурласан TF-IDF
representation хэлбэрээр төлөөлсөн. Unigram болон bigram feature-үүдийг
ашиглаж, үг бүрийн ач холбогдлыг TF-IDF жин ашиглан тодорхойлсон. Үүний
дараа өндөр хэмжээст feature space-ийг багасгах зорилгоор Principal
Component Analysis (PCA) хэрэглэсэн бөгөөд PCA-ийн гаргасан
компонентуудыг document-ийн эцсийн representation гэж үзсэн.

Hyperparameter-ууд:TF-IDF-ийн хувьд unigram болон bigram ашигласан.
PCA-д variance-д суурилан компонентуудын тоог сонгосон боловч яг хэдэн
компонент ашигласныг нарийвчлан заагаагүй. SVM загварт kernel болон C
параметрийн тодорхой утгыг дурдаагүй. KNN загварт k-ийн утгыг туршилтаар
сонгосон гэж тайлбарласан ч яг тоон утгыг өгүүлээгүй. Learning rate,
epoch зэрэг deep learning-ийн hyperparameter ашиглагдаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
ашиглаагүй. Machine Learning талд Support Vector Machine болон K-Nearest
Neighbour алгоритмуудыг ашигласан. SVM-ийг margin-д суурилсан ангилагч
байдлаар, KNN-ийг зайд суурилсан ангилагч байдлаар ашиглаж, эдгээрийн
гүйцэтгэлийг хооронд нь харьцуулсан.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
үндсэн preprocessing алхмууд болох tokenization, stopword removal,
stemming ашигласан. Feature engineering-д TF-IDF, dimensionality
reduction-д PCA, supervised learning-д SVM болон KNN ашигласан. Мөн
classification result-ийг confusion matrix-ээр тайлбарласан.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон accuracy ашиглан
үнэлсэн бөгөөд recall-ийг туслах үзүүлэлт байдлаар дурдсан. Precision,
F1-score-ийг үндсэн metric болгон ашиглаагүй. IMDB Large Movie Review
Dataset дээр хийсэн туршилтаар SVM загвар ойролцоогоор 81.7% accuracy
үзүүлсэн бол KNN загварын accuracy ойролцоогоор 65.1% байсан. PCA +
TF-IDF + SVM хослол нь хамгийн сайн гүйцэтгэл үзүүлсэн гэж дүгнэсэн.

2.4 [Addressing Sentiment Analysis Challenges within AI Media Platform:
The Enabling Role of an AI Powered
Chatbot](https://rce.feaa.ugal.ro/stories/RCE2021/AvramRusu.pdf)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Avram болон Rusu**** нар
****2021 онд**** хийж, ****Applied Sciences (MDPI)**** сэтгүүлд
нийтэлсэн. Судалгаа нь бодит хэрэглээний орчинд, тухайлбал AI-д
суурилсан chatbot систем дотор sentiment analysis-ийг хэрэгжүүлэхэд
тулгардаг сорилтуудыг шийдэхэд чиглэсэн. Зорилго нь олон эх сурвалжтай,
их хэмжээний текст өгөгдөл дээр хурд болон нарийвчлалын хувьд
тохиромжтой загвар сонгох бөгөөд үүний тулд fastText болон BERT
загваруудын гүйцэтгэлийг харьцуулан судалсан.

Embedding / representation learning арга: Судалгаанд хоёр төрлийн
embedding ашигласан. Эхнийх нь fastText-ийн subword-д суурилсан word
embedding бөгөөд үгсийн дотоод бүтэц, бичгийн алдааг тодорхой хэмжээнд
харгалзан үздэг. Хоёр дахь нь Transformer архитектурт суурилсан BERT-ийн
contextual embedding бөгөөд өгүүлбэр доторх үгсийн харилцан хамаарлыг
хоёр чиглэлд тооцож сургадаг. fastText embedding нь classifier-тэй хамт
сургагдсан бол BERT-ийн хувьд pretrained загварыг fine-tuning хийж
ашигласан.

Hyperparameter-ууд: fastText загварт сургалтын epoch-ийн тоо 7 байсан
бөгөөд CPU орчинд сургаж туршсан. BERT загварт fine-tuning хийхдээ 1
epoch сургалт хийж, GPU ашигласан. Dataset-ийг 90 хувь сургалт, 10 хувь
тест болгон хуваасан. Learning rate, batch size зэрэг нарийвчилсан
hyperparameter-уудыг paper-т тодорхой дурдсангүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
талд fastText neural text classifier болон BERT (Transformer-based deep
neural network) ашигласан. Уламжлалт machine learning алгоритм
ашиглаагүй. Судалгаа нь цэвэр deep learning загваруудын бодит систем дэх
гүйцэтгэлийг үнэлэхэд чиглэсэн.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
preprocessing алхмууд, supervised learning, transfer learning,
fine-tuning аргачлалуудыг ашигласан. Том хэмжээний өгөгдөл боловсруулах,
chatbot системд интеграцлах программ хангамжийн түвшний архитектурын
шийдлүүдийг мөн авч үзсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг accuracy, precision, recall,
F1-score ашиглан үнэлсэн. IMDB movie review dataset дээр fastText загвар
ойролцоогоор 88.1% accuracy үзүүлсэн бол BERT загвар 94.6--95.3%
accuracy хүрсэн. Amazon-ийн том хэмжээний review dataset дээр BERT
загварын accuracy ойролцоогоор 95% байсан бол fastText-ийн гүйцэтгэл
харьцангуй доогуур гарсан. Imbalanced өгөгдөлтэй earphones review
dataset дээр fastText 73.7% accuracy үзүүлсэн бол BERT ойролцоогоор 90%
accuracy хүрч, contextual embedding нь бодит хэрэглээний орчинд илүү
тогтвортой болохыг харуулсан.

2.5 [Adaptive Rider Feedback Artificial Tree Optimization-Based Deep
Neuro-Fuzzy Network for Classification of Sentiment
Grade](../../Downloads/Adaptive_Rider_Feedback_Artificial_-1.pdf)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Jasti болон Raj Kumar****
нар ****2023 онд**** хийж, ****Journal of Telecommunications and
Information Technology**** сэтгүүлд нийтэлсэн. Судалгаа нь sentiment
analysis-ийг зөвхөн эерэг, сөрөг гэж ангилахаас гадна sentiment-ийн
нарийвчилсан түвшин буюу sentiment grade тодорхойлоход чиглэсэн.
Зохиогчид гар аргаар гаргасан олон төрлийн текстийн шинжүүдийг
нейро-фаззи загварт оруулж, сургалтын параметрүүдийг meta-heuristic
аргаар оновчлох замаар гүйцэтгэлийг сайжруулах зорилго тавьсан.

Embedding / representation learning арга: Neural word embedding
ашиглаагүй. Текстийг feature-д суурилсан representation хэлбэрээр
төлөөлсөн. Үүнд SentiWordNet-д суурилсан эерэг, сөрөг утгын оноо,
TF-IDF-ийн статистик шинжүүд, emoticon-д суурилсан шинж, spam болон
pronoun-д суурилсан шинжүүдийг ашигласан. Эдгээр feature-үүдийг нэгтгэн
нейро-фаззи сүлжээний оролт болгон ашигласан.

Hyperparameter-ууд: Deep Neuro-Fuzzy Network-д bell-shaped membership
function ашигласан. Hidden layer-ийн тоо болон fuzzy rule-ийн бүтэц нь
dataset-ээс хамааран өөрчлөгдсөн. Optimization хэсэгт Adaptive Rider
Feedback Artificial Tree Optimization алгоритмын rider-ийн хурд, чиглэл,
жин зэрэг параметрүүдийг ашигласан. Learning rate, batch size, epoch
зэрэг уламжлалт deep learning hyperparameter-уудыг тодорхой дурдaагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
талд Deep Neuro-Fuzzy Network ашигласан. Уламжлалт gradient-д суурилсан
optimizer-ийн оронд population-based meta-heuristic optimization (RFATO)
ашиглаж, сүлжээний параметрүүдийг оновчилсон. Classical machine learning
загваруудыг үндсэн classifier байдлаар ашиглаагүй, харин baseline болгон
зарим deep болон hybrid загваруудтай харьцуулсан.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
preprocessing, feature extraction, fuzzy logic, neural network,
meta-heuristic optimization, supervised learning аргачлалуудыг
ашигласан. Мөн decision-making системд ашиглагддаг angular similarity
болон feature fusion ойлголтуудыг хэрэглэсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг accuracy, precision,
sensitivity (recall), specificity ашиглан үнэлсэн. F1-score-ийг үндсэн
metric болгон ашиглаагүй. IMDB Large Movie Review Dataset дээр санал
болгосон арга ойролцоогоор 97.2% accuracy үзүүлсэн. Datafiniti Product
Review dataset дээр accuracy ойролцоогоор 99.0% хүрсэн бол Amazon review
dataset дээр 95.8% орчим accuracy гарсан. Эдгээр үр дүн нь CNN, RNN
болон бусад baseline deep learning загваруудаас тогтвортойгоор өндөр
байсан гэж тайлагнасан.

2.6 [Baselines and Bigrams: Simple, Good Sentiment and Topic
Classification](https://aclanthology.org/P12-2018.pdf)

Товч танилцуулга Энэхүү судалгааны ажлыг ****Sida Wang болон Christopher
D. Manning**** нар ****2012 онд**** хийж, ****ACL (Association for
Computational Linguistics)****-ийн бага хуралд нийтэлсэн. Судалгааны гол
зорилго нь sentiment болон topic classification-д өргөн хэрэглэгддэг
Naive Bayes болон Support Vector Machine зэрэг энгийн baseline
загваруудыг системтэйгээр үнэлж, зөв тохируулсан тохиолдолд эдгээр
загварууд нь илүү төвөгтэй аргачлалуудтай өрсөлдөхүйц гүйцэтгэл үзүүлж
чаддагийг харуулах явдал байв. Судалгаа нь "baseline"-ийг дутуу үнэлэх
хандлагыг шүүмжилсэн онцлогтой.

Embedding / representation learning арга: Neural embedding ашиглаагүй.
Текстийг Bag-of-Words загварт тулгуурласан unigram болон bigram
feature-үүдээр төлөөлсөн. Үүнээс гадна Naive Bayes-ийн log-count ratio-д
суурилсан NBSVM representation ашигласан бөгөөд энэ нь generative болон
discriminative аргуудыг хослуулсан feature transformation юм. Word2Vec,
GloVe зэрэг pretrained embedding ашиглаагүй.

Hyperparameter-ууд: Naive Bayes загварт Laplace smoothing параметр α = 1
ашигласан. Linear SVM-д regularization параметр C = 1 (зарим туршилтад
0.1) хэрэглэсэн. NBSVM-д Naive Bayes болон SVM-ийн жинг хослуулах
interpolation параметр β = 0.25 байсан. Feature-үүдийг binarized (0/1)
хэлбэрээр ашигласан. Learning rate, epoch зэрэг deep learning
hyperparameter ашиглаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
ашиглаагүй. Machine Learning талд Multinomial Naive Bayes, Linear
Support Vector Machine болон эдгээрийг хослуулсан NBSVM загварыг
ашигласан. Эдгээр нь бүгд supervised learning-д суурилсан ангилагчид юм.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
preprocessing, Bag-of-Words feature extraction, n-gram modeling,
probabilistic modeling, margin-based classification, supervised learning
ашигласан. Мөн statistical significance test ашиглан үр дүнгийн
найдвартай байдлыг шалгасан.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон accuracy ашиглан
үнэлсэн. Precision, recall, F1-score-ийг үндсэн metric болгон
ашиглаагүй. Богино тексттэй sentiment dataset-үүд дээр Naive Bayes
загвар SVM-ээс илүү гүйцэтгэл үзүүлсэн бол урт review бүхий IMDB dataset
дээр Linear SVM нь Naive Bayes-ээс давсан. IMDB Large Movie Review
Dataset дээр NBSVM загвар ойролцоогоор 91--92% accuracy үзүүлж, энгийн
Bag-of-Words + SVM болон Naive Bayes-ийг хоёуланг нь давсан. Судалгаанд
unigram дээр bigram нэмэх нь бүх dataset дээр тогтвортой сайжруулалт
өгдөгийг харуулсан.

2.7 [DOMAIN ADAPTABLE MODEL FOR SENTIMENT
ANALYSIS](https://web.archive.org/web/20220707051315id_/https://www.actapress.com/pdfviewer.aspx?paperID=54999)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Kalra, Agrawal, Sharma****
нар ****2022 онд**** хийж, ****Mechatronic Systems and Control****
сэтгүүлд нийтэлсэн. Судалгааны зорилго нь sentiment analysis-д тулгардаг
хамгийн том бэрхшээлүүдийн нэг болох domain shift буюу нэг домэйн дээр
сурсан загварыг өөр домэйнд шууд ашиглахад гүйцэтгэл огцом буурах
асуудлыг шийдэх явдал юм. Үүний тулд зохиогчид label-тэй өгөгдөл цөөн
эсвэл байхгүй нөхцөлд ажиллах боломжтой domain adaptable hybrid загвар
санал болгосон.

Embedding / representation learning арга: Neural embedding ашиглаагүй.
Текстийг TF-IDF-д суурилсан Bag-of-Words representation хэлбэрээр
төлөөлсөн. Үүн дээр lexicon-based sentiment feature-үүдийг нэмсэн бөгөөд
VADER болон SentiWordNet-оос гаргасан sentiment score-уудыг ашигласан.
Мөн domain-ийн онцлогийг барих зорилгоор topic-aware representation
ашиглаж, текстийг урьдчилан topic-оор ангилсны дараа sentiment
classifier-д оруулсан.

Hyperparameter-ууд: TF-IDF feature ашигласан. Topic classification-д 6
сэдэв (Business, Sports, Health, Entertainment, Politics, Technology)
тогтмол сонгосон. Sentiment classifier-д SVM ашигласан боловч kernel
болон C параметрийн тодорхой утгыг paper-т дурдаагүй. Supervised болон
unsupervised горимд шилжих босгыг label-тэй өгөгдлийн тоо 1000-аас их
эсэхээр тодорхойлсон. Learning rate, epoch зэрэг deep learning
hyperparameter ашиглаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
ашиглаагүй. Machine Learning талд Multinomial Naive Bayes-ийг topic
classification-д, Support Vector Machine-ийг sentiment classification-д
ашигласан. Label-тэй өгөгдөл хангалтгүй үед lexicon-based unsupervised
sentiment analysis ашиглаж, хангалттай үед supervised SVM рүү шилжих
hybrid арга хэрэгжүүлсэн.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
preprocessing, feature extraction, topic modeling, supervised болон
unsupervised learning, transductive learning аргачлалуудыг ашигласан.
Мөн cross-domain evaluation хийж, домэйн хоорондын шилжилтийн нөлөөг
судалсан.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг accuracy, precision, recall,
F1-score ашиглан үнэлсэн. SemEval-2013 cross-domain dataset дээр дундаж
accuracy ойролцоогоор 83% байсан. IMDB dataset дээр micro-average
F1-score ойролцоогоор 0.78, macro-average F1-score ойролцоогоор 0.77
гарсан. Amazon болон Twitter-д суурилсан dataset-үүд дээр санал болгосон
hybrid загвар нь зарим deep learning загваруудтай ойролцоо, эсвэл давсан
гүйцэтгэл үзүүлсэн гэж тайлагнасан. Судалгаанд domain-adaptive хандлага
нь шинэ домэйнд гүйцэтгэлийг тогтвортой хадгалах давуу талтай болохыг
харуулсан.

2.8 [Distributed Representations of Sentences and
Documents](https://proceedings.mlr.press/v32/le14.html)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Quoc Le болон Tomas
Mikolov**** нар ****2014 онд**** хийж, ****International Conference on
Machine Learning (ICML)****-д нийтэлсэн. Судалгааны зорилго нь өгүүлбэр
болон баримт бичгийг тогтмол хэмжээтэй dense vector хэлбэрээр төлөөлөх
шинэ representation learning арга боловсруулах явдал юм. Уламжлалт
Bag-of-Words болон TF-IDF загварууд нь үгсийн дараалал болон семантик
хамаарлыг бүрэн илэрхийлж чаддаггүй тул зохиогчид document-level
embedding сургах Paragraph Vector аргыг санал болгосон.

Embedding / representation learning арга: Судалгаанд unsupervised
document embedding ашигласан. Paragraph Vector буюу Doc2Vec аргын хоёр
хувилбарыг ашигласан. Эхнийх нь Distributed Memory (PV-DM) бөгөөд
document vector болон тухайн орчны үгсийг хамтад нь ашиглан дараагийн
үгийг таамагладаг. Хоёр дахь нь Distributed Bag of Words (PV-DBOW)
бөгөөд зөвхөн document vector ашиглан тухайн баримт бичигт орших үгсийг
таамагладаг. Эдгээр document vector-уудыг тухайн баримт бичгийн эцсийн
embedding гэж үзсэн. Word2Vec-ийн skip-gram, CBOW ойлголтод суурилсан
боловч document-level representation сургадаг онцлогтой.

Hyperparameter-ууд: Document болон word vector-ийн хэмжээ 100-аас
400-ийн хооронд байсан. Context window-ийн хэмжээ 5--10. Negative
sampling-ийн тоо 5--15. Training epoch-ийн тоо 10--20. Learning rate
эхэндээ 0.025 орчим байж, сургалтын явцад багасгасан. Нэг hidden
layer-тэй shallow neural network ашигласан.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
талд shallow neural language model ашигласан. Сурсан document
embedding-ийг дараагийн шатанд Machine Learning загваруудад ашигласан
бөгөөд Logistic Regression, Support Vector Machine зэрэг шугаман
ангилагчид дээр sentiment болон topic classification хийсэн. Deep
sequence model (LSTM, RNN) ашиглаагүй.

Ашиглагдсан Computer Science аргууд: Distributional semantics,
unsupervised representation learning, neural language modeling, negative
sampling, hierarchical softmax ашигласан. Мөн document classification-д
supervised learning ашиглан embedding-ийн чанарыг үнэлсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон accuracy болон error
rate ашиглан үнэлсэн. Precision, recall, F1-score-ийг үндсэн metric
болгон ашиглаагүй. IMDB Large Movie Review Dataset дээр Paragraph Vector
(ялангуяа PV-DM болон PV-DBOW-ийг хослуулсан хувилбар) нь Bag-of-Words,
TF-IDF болон averaged Word2Vec representation-уудаас тогтвортойгоор
өндөр accuracy үзүүлсэн. Судалгаанд Doc2Vec ашигласан загварууд тухайн
үеийн state-of-the-art үр дүнд хүрсэн гэж тайлагнасан бөгөөд
document-level embedding нь sentiment analysis болон topic
classification-д үр дүнтэй болохыг баталсан.

2.9 [SENTIMENT ANALYSIS FOR MOVIEREVIEWS USING ARTIFICIAL NEURAL
NETWORKS AND RECURRENT NEURAL
NETWORKS](https://rajshree.ac.in/wp-content/uploads/2023/10/AqibCS-23.pdf)

Энэхүү судалгааны ажил нь ****Aqib Akhtar Zia****-ийн ****2023 онд****
гүйцэтгэсэн ****магистрын түвшний (M.Tech) төгсөлтийн ажил**** бөгөөд
****их сургуулийн судалгааны орчинд**** хийгдсэн. Судалгааны зорилго нь
киноны review-үүдийн sentiment analysis-д уламжлалт машин сургалтын
аргууд болон deep learning-д суурилсан нейрон сүлжээний загваруудын
гүйцэтгэлийг ижил нөхцөлд харьцуулан үнэлэх явдал юм. Ялангуяа feature
engineering-д суурилсан classical ML загварууд болон representation
learning-д суурилсан ANN, RNN, LSTM загваруудын ялгааг туршилтаар
харуулахыг зорьсон.

Embedding / representation learning арга: Судалгаанд хоёр өөр
representation ашигласан. Эхнийх нь classical feature-based
representation бөгөөд Bag-of-Words, n-gram болон TF-IDF-д суурилсан
үгийн давтамжийн шинжүүдийг ашигласан. Хоёр дахь нь neural embedding
бөгөөд trainable word embedding layer ашиглан үг бүрийг dense vector
хэлбэрээр төлөөлсөн. LSTM болон RNN загваруудын hidden state-ийг тухайн
review-ийн sequence-based representation гэж үзсэн. Pretrained embedding
(Word2Vec, GloVe, BERT) ашиглаагүй.

Hyperparameter-ууд: Vocabulary size нь dataset-д суурилан тогтоогдсон.
Neural embedding-ийн хэмжээ ойролцоогоор 100. LSTM загварт 128 hidden
unit ашигласан. Dropout layer хэрэглэсэн. Optimizer нь Adam, loss
function нь binary cross-entropy байсан. Epoch, batch size, learning
rate-ийн нарийвчилсан утгуудыг бүх загварт ижил түвшинд тохируулж
туршсан боловч яг тоон утгыг thesis-д бүрэн стандартчилж тайлагнаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Machine Learning
талд Multinomial Naive Bayes, Logistic Regression, Linear Support Vector
Machine, XGBoost ашигласан. Deep Learning талд Artificial Neural Network
(feedforward MLP), Recurrent Neural Network, Long Short-Term Memory
(LSTM) загваруудыг ашигласан. Судалгаанд classical ML болон DL
загваруудыг нэг ижил dataset дээр шууд харьцуулсан нь онцлог.

Ашиглагдсан Computer Science аргууд: Natural Language Processing-ийн
preprocessing, tokenization, stopword removal, feature extraction,
supervised learning, backpropagation, sequence modeling, gradient-based
optimization ашигласан. Мөн confusion matrix-д суурилсан үнэлгээ хийсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон accuracy ашиглан
үнэлсэн бөгөөд confusion matrix ашиглан ангиллын чанарыг тайлбарласан.
Precision, recall, F1-score-ийг хоёрдогч байдлаар дурдсан боловч үндсэн
metric болгон ашиглаагүй. IMDB Large Movie Review Dataset дээр Linear
SVM загвар ойролцоогоор 89.6% accuracy үзүүлсэн нь classical ML
загварууд дундаа хамгийн өндөр байсан. Deep learning загваруудаас LSTM
нь ANN болон энгийн RNN-ээс илүү тогтвортой үр дүн үзүүлсэн боловч
тухайн туршилтын нөхцөлд Linear SVM-ийг давж чадаагүй. Судалгааны
дүгнэлтээр зөв тохируулсан classical ML загварууд зарим тохиолдолд deep
learning загваруудаас илүү гүйцэтгэл үзүүлж болохыг харуулсан.

2.10 [Analysis of IMDB Reviews For Movies And Television Series using
SAS® Enterprise Miner™ and SAS® Sentiment Analysis
Studio](https://business.okstate.edu/site-files/archive/docs/analytics/11001-2016.pdf)

Товч танилцуулга: Энэхүү судалгааны ажлыг ****Jadhavar, Pawar, Patil****
нар ****2016 онд**** хийж, ****SAS Global Forum 2016****-д танилцуулсан.
Судалгаа нь кино болон телевизийн цувралын IMDB review-үүд дээр
sentiment analysis-ийг ****аж үйлдвэрийн зориулалттай аналитик программ
хангамж**** ашиглан хэрэгжүүлэхэд чиглэсэн. Зорилго нь deep learning
загвар хөгжүүлэхээс илүүтэйгээр текст mining, rule-based болон статистик
аргачлалуудыг ашиглан бодит бизнесийн орчинд хэрэгжих боломжтой
sentiment analysis систем боловсруулах явдал байв.

Embedding / representation learning арга: Neural embedding ашиглаагүй.
Текстийг Bag-of-Words-д суурилсан feature representation хэлбэрээр
төлөөлсөн. SAS Enterprise Miner-ийн text parsing болон text filtering
модулиудыг ашиглан үгийн давтамж, synonym grouping, term frequency-д
суурилсан шинжүүдийг гаргасан. Үгсийн ач холбогдлыг Smoothed Relative
Frequency болон Chi-square статистик аргуудаар жинлэсэн.

Hyperparameter-ууд: Text clustering-д Expectation--Maximization алгоритм
ашиглаж, кластерын тоог 10 болгон тохируулсан. Topic modeling-д 7 topic
ашигласан. Statistical sentiment model-д сургалт болон тестийн өгөгдлийг
80/20 харьцаагаар хуваасан бол rule-based загварт 70/30 харьцаа
хэрэглэсэн. Learning rate, epoch зэрэг deep learning hyperparameter
ашиглаагүй.

Ашигласан Deep Learning болон Machine Learning аргууд: Deep Learning
ашиглаагүй. Machine Learning талд статистик загварчлалд суурилсан
sentiment classifier ашигласан. Мөн rule-based sentiment analysis аргыг
SAS Sentiment Analysis Studio-ийн Text Rule Builder ашиглан
хэрэгжүүлсэн. Эдгээрийг baseline болон харьцуулалтын зорилгоор
ашигласан.

Ашиглагдсан Computer Science аргууд: Text mining, information retrieval,
statistical feature weighting, clustering, topic modeling, rule-based
reasoning, supervised learning ашигласан. Мөн concept link analysis
ашиглан үгсийн хоорондын хамаарлыг тайлбарлах боломжийг бүрдүүлсэн.

Үр дүн, ашигласан үнэлгээний арга: Үр дүнг голчлон precision болон
accuracy ашиглан үнэлсэн. F1-score, recall-ийг үндсэн metric болгон
ашиглаагүй. Statistical sentiment model нь ойролцоогоор 78.4% precision
үзүүлсэн бол rule-based sentiment model нь validation өгөгдөл дээр 80.8%
precision хүрсэн. Нэмэлтээр 1000 review дээр хийсэн scoring туршилтаар
нийт accuracy ойролцоогоор 79--80% орчим байсан. Судалгаанд rule-based
арга нь илүү тайлбарлагдахуйц, практик хэрэглээнд тохиромжтой болохыг
онцолсон.

2.11 10 судалгааны ажлын харьцуулсан хүснэгт

----------------------------------------
  | №  | Судалгааны ажил                   | Embedding / Representation | DL / ML аргууд | CS аргууд           | Metric   | Гол үр дүн |
| -- | --------------------------------- | -------------------------- | -------------- | ------------------- | -------- | ---------- |
| 1  | Semisupervised Autoencoder (2016) | BoW + Autoencoder          | AE, SVM        | Semi-supervised     | Accuracy | ~93.8%     |
| 2  | LSTM (2020)                       | Trainable embedding        | LSTM           | Sequence modeling   | Accuracy | ~99.5%     |
| 3  | ML Approach (2022)                | TF-IDF + PCA               | SVM, KNN       | Feature engineering | Accuracy | 81.7%      |
| 4  | Chatbot (2021)                    | fastText, BERT             | BERT           | Transfer learning   | F1       | ~95%       |
| 5  | Neuro-Fuzzy (2023)                | Feature-based              | Neuro-Fuzzy    | Optimization        | Accuracy | ~97.2%     |
| 6  | Baselines (2012)                  | BoW                        | NB, SVM        | n-gram              | Accuracy | ~92%       |
| 7  | Domain Adaptable (2022)           | TF-IDF + Lexicon           | SVM            | Topic modeling      | F1       | ~0.78      |
| 8  | Doc2Vec (2014)                    | Doc2Vec                    | SVM            | Representation      | Accuracy | SOTA       |
| 9  | ANN & RNN (2023)                  | TF-IDF, Embedding          | ANN, LSTM      | Feature engineering | Accuracy | ~89.6%     |
| 10 | SAS (2016)                        | BoW                        | Statistical    | Rule-based          | Accuracy | ~80%       |


Энэхүү судалгаанд IMDB Large Movie Review Dataset-ийг ашигласан
sentiment analysis чиглэлийн арван судалгааны ажлыг хамруулан шинжиллээ.
Судалгаануудын харьцуулалтаас харахад IMDB dataset нь тэнцвэртэй бүтэц,
хангалттай хэмжээтэй өгөгдлөөрөө sentiment analysis судалгаанд стандарт
benchmark болж тогтсон байна.

Уламжлалт машин сургалтын аргууд болох Bag-of-Words болон TF-IDF-д
суурилсан SVM, Naive Bayes загварууд нь энгийн боловч тогтвортой
гүйцэтгэл үзүүлж, IMDB dataset дээр 80--92 хувийн accuracy хүргэж
чадсан. Representation learning-д суурилсан autoencoder болон Doc2Vec
аргууд нь feature engineering-ээс хараат бус embedding сургах боломж
олгосон ч гүйцэтгэлийн сайжруулалт харьцангуй хязгаарлагдмал байв.

Гүн сургалтын загварууд болох LSTM болон BERT нь текстийн дараалал,
контекстийн хамаарлыг илүү сайн загварчилж, хамгийн өндөр гүйцэтгэлийг
үзүүлсэн. Ялангуяа BERT зэрэг contextual embedding-д суурилсан загварууд
нь олон домэйн болон өгөгдлийн тэнцвэр алдагдсан нөхцөлд илүү тогтвортой
ажиллаж байгааг харуулсан.

Мөн hybrid болон domain-adaptive аргууд нь label-тэй өгөгдөл
хязгаарлагдмал үед sentiment analysis хийх боломжийг өргөжүүлж байгааг
харуулсан ч загварын нарийн төвөгтэй байдал нь практик хэрэглээнд
тодорхой хязгаарлалт үүсгэж байна. Ерөнхийд нь дүгнэвэл, судалгааны
зорилгоос хамааран classical ML, deep learning болон hybrid аргуудын аль
аль нь өөрийн давуу талтай бөгөөд цаашдын судалгаанд эдгээрийг
уялдуулсан шийдэл чухал ач холбогдолтой гэж үзэж байна.

3. EMBEDDING АРГУУД БОЛОН ОНОЛЫН ҮНДЭС

3.1 TF-IDF

TF-IDF (Term Frequency--Inverse Document Frequency) нь текстийг тоон
хэлбэрт оруулах хамгийн эртний, статистикт суурилсан арга юм. Энэхүү
ойлголтыг анх Karen Spärck Jones 1972 онд мэдээлэл хайлтын салбарт
танилцуулсан. TF-IDF нь тухайн үг нэг баримт бичигт хэр давтамжтай гарч
байгааг болон нийт баримтуудын дунд хэр ховор байгааг хамтад нь тооцож,
тухайн үгийн ач холбогдлыг тодорхойлдог. Математик байдлаар TF-IDF нь
дараах байдлаар тодорхойлогдоно. TF(t,d)=т ухайн t үг баримт d дотор
гарсан давтамж IDF(t) = log ( N / df(t) ) энд N нь нийт баримтын тоо,
df(t) нь t үг орсон баримтын тоо юм. TF-IDF(t, d) = TF(t, d) × IDF(t) .
Энэ арга нь үгийн дараалал болон утгын хамаарлыг тооцдоггүй, зөвхөн
статистик жинд суурилдаг тул classical machine learning загваруудад
өргөн хэрэглэгддэг.

3.2 Word2Vec -- CBOW

Word2Vec-ийг Tomas Mikolov болон хамтрагчид 2013 онд Google-ийн
судалгааны хүрээнд танилцуулсан. CBOW (Continuous Bag of Words) загвар
нь үгсийн орчныг ашиглан төв үгийг таамаглах зарчимд суурилдаг. CBOW
загварын зорилго нь дараах магадлалыг ихэсгэх явдал юм. max ∑ log p(wₜ
\| wₜ₋ₖ , ... , wₜ₋₁ , wₜ₊₁ , ... , wₜ₊ₖ) Энд wₜ нь таамаглагдах төв үг,
харин бусад нь түүний орчны үгс юм. Практикт softmax тооцооллын зардлыг
бууруулахын тулд negative sampling ашигладаг. CBOW нь их хэмжээний
өгөгдөл дээр хурдан, тогтвортой сурдаг онцлогтой.

3.3 Word2Vec -- Skip-gram

Skip-gram нь мөн 2013 онд Mikolov нарын санал болгосон Word2Vec-ийн хоёр
дахь архитектур юм. Skip-gram нь CBOW-ийн эсрэг логиктой бөгөөд нэг төв
үгийг өгөөд түүний орчны үгсийг таамагладаг. Зорилтот функц нь: max ∑ₜ
∑\_{c∈context(t)} log p(w_c \| wₜ) Энэхүү загвар нь ховор үгсийн
representation-ийг илүү сайн сурдаг тул жижиг dataset дээр CBOW-оос
давуу талтай байдаг. Skip-gram мөн negative sampling болон hierarchical
softmax-ыг ашиглан сургагддаг.

3.4 BERT (Base BERT)

BERT (Bidirectional Encoder Representations from Transformers)-ийг Jacob
Devlin болон хамтрагчид 2018 онд Google Research-т боловсруулсан. BERT
нь Transformer encoder архитектурт суурилсан бөгөөд өгүүлбэрийг хоёр
талаас нь зэрэг харж сурдгаараа өмнөх embedding аргуудаас ялгаатай.
Self-attention механизмын үндсэн томьёо нь: Attention(Q, K, V) =
softmax( QKᵀ / √dₖ ) V BERT-ийн үндсэн сургалтын зорилго нь Masked
Language Modeling бөгөөд өгүүлбэрийн зарим токеныг нууж, зөв үгийг
таамаглах cross-entropy loss-ийг багасгахад чиглэнэ. Base BERT нь 12
encoder layer, 768 hidden size, 12 attention head-тэй стандарт хувилбар
юм.
.5 RoBERTa

RoBERTa-г Yinhan Liu болон хамтрагчид 2019 онд Facebook AI-д
танилцуулсан. RoBERTa нь BERT-ийн архитектурыг өөрчлөлгүйгээр сургалтын
стратегийг сайжруулсан хувилбар юм. Үүнд илүү их өгөгдөл, илүү удаан
сургалт, dynamic masking, Next Sentence Prediction-ийг хассан зэрэг
өөрчлөлтүүд ордог. Математик бүтэц нь BERT-тэй ижил бөгөөд Transformer
encoder болон self-attention-д суурилдаг.

3.6 ALBERT

ALBERT (A Lite BERT)-ийг Zhenzhong Lan болон хамтрагчид 2019 онд
танилцуулсан. Гол зорилго нь BERT-ийн параметрийн тоог багасгаж,
тооцооллын үр ашиг нэмэгдүүлэх явдал юм. ALBERT нь хоёр үндсэн санааг
ашигладаг. Нэгд, embedding factorization буюу том vocabulary
embedding-ийг бага хэмжээст орон зайд хуваах. Хоёрт, Transformer
layer-үүдийн параметрийг хооронд нь хуваалцах. Үүний үр дүнд BERT-тэй
ойролцоо гүйцэтгэлтэй боловч илүү хөнгөн загвар бий болсон.

3.7 HateBERT

HateBERT-ийг Tommaso Caselli болон хамтрагчид 2020 онд танилцуулсан.
HateBERT нь BERT-ийн архитектурт суурилсан боловч hate speech, abusive
language ихтэй домэйн өгөгдөл дээр дахин pretraining хийсэн
domain-specific загвар юм. Математик үндэс нь BERT-тэй ижил бөгөөд ялгаа
нь сургалтын өгөгдөлд оршдог. Ийм domain-adaptive pretraining нь
тодорхой сэдэвт хамаарах хэллэгийг илүү нарийвчлалтай ойлгох боломж
олгодог.

3.8 SBERT (Sentence-BERT)

Sentence-BERT-ийг Nils Reimers болон Iryna Gurevych нар 2019 онд санал
болгосон. SBERT нь BERT-ийг өгүүлбэр түвшний embedding гаргахад
тохируулсан хувилбар юм. SBERT нь Siamese network бүтэц ашиглан өгүүлбэр
бүрийг тусад нь encoding хийж, cosine similarity-д суурилсан төстэй
байдлыг тооцдог. cos(u, v) = (u · v) / (\|\|u\|\| \|\|v\|\|) Сургалтын
явцад өгүүлбэр хосуудын төстэй байдлыг алдааны функцээр оновчлон,
тогтмол хэмжээтэй sentence embedding сургадаг.

4. ТУРШИЛТЫН ОРЧИН

Энэхүү судалгааны туршилтуудыг дараах техник хангамж болон програм
хангамжийн орчинд гүйцэтгэсэн. Туршилтыг ****зөөврийн компьютер
(laptop)**** дээр гүйцэтгэсэн бөгөөд техник үзүүлэлтүүд нь дараах
байдалтай.

-   ****Процессор (CPU):**** Intel Core i5, 12-р үе

-   ****Санах ой (RAM):**** 16 GB

-   ****График карт (GPU):**** NVIDIA GeForce RTX 4050

-   ****Үйлдлийн систем:**** Linux (Ubuntu суурьтай орчин)

-   ****Python хувилбар:**** Python 3.12

    ****Гол сангууд:****

    -   PyTorch
    -   Transformers (HuggingFace)
    -   Scikit-learn
    -   Gensim
    -   SciPy, NumPy, Pandas

Transformer суурьтай embedding (BERT, SBERT, ALBERT, HateBERT, RoBERTa)
үүсгэх үед GPU-г ашигласан бол, харин TF-IDF болон Word2Vec суурьтай
загварууд нь CPU дээр гүйцэтгэсэн.

5. Фолдерийн бүтэц (Project Folder Structure)

Туршилтын бүх үр дүн, embedding болон загварын файлуудыг ****нэгдсэн,
системтэй бүтэцтэйгээр**** хадгалсан. Үндсэн бүтэц нь дараах байдалтай.

![Зураг 2: Фолдерын ерөнхий
бүтэц](./100000010000011E000000D6E89BF908.png "fig:"){width="2.9791in"
height="2.1673in"}![Зураг 1: Үр дүнгийн хадгалагдсан байдал
](./10000001000000F7000000B665649862.png "fig:"){width="2.9327in"
height="2.161in"}

Туршилтын явцад ашигласан өгөгдөл болон завсрын үр дүнг системтэйгээр
хадгалах зорилгоор төслийн фолдерийн бүтцийг нэгдсэн байдлаар зохион
байгуулсан. Цэвэрлэсэн өгөгдөл нь data фолдерт хадгалагдсан бөгөөд
embedding-үүд, загварын параметрүүд, cross-validation болон эцсийн
үнэлгээний үр дүнг artifacts фолдерт тусад нь хадгалсан. Embedding арга
бүрийн хувьд тусгай дэд фолдер үүсгэж, тухайн аргын cross-validation-ийн
үр дүн, сонгогдсон параметрүүд болон эцсийн туршилтын CSV файлуудыг
ялгаж байршуулсан. Ийнхүү фолдерийн бүтцийг стандартчилснаар туршилтыг
дахин давтах, үр дүнг баталгаажуулах болон embedding аргуудын хооронд
харьцуулалт хийхэд хялбар болсон.

Загварын гиперпараметрийг оновчтой сонгохын тулд cross-validation аргыг
ашигласан. Тодруулбал, Logistic Regression загварт Stratified 5-fold
cross-validation хэрэгжүүлж, сургалтын өгөгдөл дэх ангиудын харьцааг
хадгалсан байдлаар өгөгдлийг хуваасан. Cross-validation-ийн үнэлгээний
шалгуураар F1-score ашигласан бөгөөд энэ нь ангиллын тэнцвэргүй
байдалтай өгөгдөл дээр загварын гүйцэтгэлийг илүү бодитой илэрхийлэх
давуу талтай. Cross-validation-ийн үр дүнд үндэслэн Logistic Regression
загварын C параметрийн олон утгуудыг шалгаж, хамгийн өндөр дундаж
F1-score авсан арван өөр параметрийн утгыг сонгон авсан.

Сонгогдсон параметрүүдийг ашиглан эцсийн туршилтуудыг гүйцэтгэсэн.
Тодруулбал, embedding арга бүрийн хувьд cross-validation-ээр
тодорхойлсон арван өөр параметрийн тохиргоог ашиглаж, Logistic
Regression, Random Forest болон AdaBoost гэсэн гурван өөр ангилагч
загварыг сургаж үнэлсэн. Ингэснээр нэг embedding аргын хувьд олон
давтамжтай туршилт хийгдэж, загваруудын гүйцэтгэлийн тогтвортой байдлыг
шалгах боломж бүрдсэн. TF-IDF embedding нь өндөр хэмжээстэй, sparse шинж
чанартай тул дараалалд суурилсан нейрон сүлжээтэй зохимжгүй гэж үзэн
LSTM загварыг ашиглаагүй бөгөөд энэ нь онолын болон практик судалгаанд
өргөн хэрэглэгддэг хандлагатай нийцнэ.

Бүх embedding аргуудыг адил өгөгдөл, ижил сургалт ба тестийн хуваалт,
нэгэн ижил cross-validation протокол болон ижил үнэлгээний хэмжүүрүүдээр
үнэлсэн. Үнэлгээнд Accuracy, Precision, Recall болон F1-score
үзүүлэлтүүдийг ашигласан бөгөөд ингэснээр embedding аргууд болон
ангилагч загваруудын хооронд шударга, дахин давтагдах боломжтой
харьцуулалт хийх нөхцөл бүрдсэн. Ийм зохион байгуулалттай туршилтын
аргачлал нь судалгааны үр дүнг бодитой, найдвартай байдлаар дүгнэх
боломжийг олгосон.

6. ҮР ДҮН

6.1 TF-IDF + ангилагч загваруудын гүйцэтгэл

| TF-IDF параметр (C) | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM |
| ------------------- | ------------------------------ | ------------------------ | ------------------- | ---- |
| 2.0                 | 0.9007 / 0.9014                | 0.8483 / 0.8465          | 0.7353 / 0.7673     | –    |
| 1.0                 | 0.8994 / 0.9002                | 0.8493 / 0.8471          | 0.7353 / 0.7673     | –    |
| 5.0                 | 0.8977 / 0.8981                | 0.8474 / 0.8455          | 0.7353 / 0.7673     | –    |
| 0.5                 | 0.8954 / 0.8966                | 0.8503 / 0.8485          | 0.7353 / 0.7673     | –    |
| 10.0                | 0.8934 / 0.8938                | 0.8499 / 0.8484          | 0.7353 / 0.7673     | –    |
| 0.1                 | 0.8809 / 0.8835                | 0.8503 / 0.8488          | 0.7353 / 0.7673     | –    |
| 0.05                | 0.8724 / 0.8755                | 0.8518 / 0.8497          | 0.7353 / 0.7673     | –    |
| 0.01                | 0.8543 / 0.8589                | 0.8507 / 0.8489          | 0.7353 / 0.7673     | –    |


TF-IDF дээр LSTM хэрэглэхэд дараах практик асуудлууд гардаг. Нэгдүгээрт,
TF-IDF векторын хэмжээ маш өндөр (олон мянган feature) бөгөөд LSTM-д
оруулахын тулд dense хэлбэрт шилжүүлэх шаардлагатай болдог. Энэ нь санах
ой болон тооцооллын хувьд үр ашиггүй бөгөөд туршилтын орчинд RAM хэт их
ашиглах эрсдэлтэй. Хоёрдугаарт, TF-IDF нь дарааллын мэдээлэл агуулаагүй
тул LSTM-ийн үндсэн давуу тал ашиглагдахгүй, улмаар загварын гүйцэтгэл
Logistic Regression зэрэг энгийн шугаман загвараас давж гарах боломж
хязгаарлагдмал болдог.

Иймээс энэхүү судалгаанд TF-IDF embedding-ийг ****зөвхөн уламжлалт
ангилагч загварууд**** (Logistic Regression, Random Forest,
AdaBoost)-тай хослуулан ашигласан бөгөөд LSTM-ийг Word2Vec болон
Transformer суурьтай embedding аргууд дээр л хэрэглэсэн.

****6.2 ****Word2Vec (CBOW) + ангилагч загваруудын гүйцэтгэл****

 | C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 1.0        | 0.8395 / 0.8411                | 0.8057 / 0.8088          | 0.7931 / 0.7951     | 0.8311 / 0.8306 |
| 5.0        | 0.8389 / 0.8406                | 0.8069 / 0.8096          | 0.7931 / 0.7951     | 0.8367 / 0.8388 |
| 2.0        | 0.8389 / 0.8406                | 0.8091 / 0.8125          | 0.7931 / 0.7951     | 0.8333 / 0.8365 |
| 10.0       | 0.8386 / 0.8403                | 0.8059 / 0.8088          | 0.7931 / 0.7951     | 0.8342 / 0.8361 |
| 0.5        | 0.8392 / 0.8409                | 0.8023 / 0.8052          | 0.7931 / 0.7951     | 0.8355 / 0.8374 |
| 0.1        | 0.8394 / 0.8413                | 0.8103 / 0.8126          | 0.7931 / 0.7951     | 0.8350 / 0.8369 |
| 0.05       | 0.8379 / 0.8401                | 0.8089 / 0.8114          | 0.7931 / 0.7951     | 0.8312 / 0.8285 |
| 0.01       | 0.8325 / 0.8348                | 0.8055 / 0.8087          | 0.7931 / 0.7951     | 0.8314 / 0.8294 |


Word2Vec-ийн CBOW embedding дээр загваруудын гүйцэтгэлийг параметрийн
өөрчлөлт бүрийн хүрээнд Accuracy болон F1-score ашиглан үнэлэв. Logistic
Regression загварын хувьд C параметрийн 0.1--1.0 орчмын утгууд дээр
хамгийн сайн гүйцэтгэл ажиглагдсан нь regularization-ийн тохиромжтой
түвшин ангиллын чанарт чухал нөлөөтэйг харуулж байна. LSTM загвар CBOW
embedding-тэй хослох үед C = 5.0 утгад хамгийн өндөр гүйцэтгэл үзүүлсэн
бөгөөд энэ нь dense, семантик төлөөлөл дараалалд суурилсан загварт үр
ашигтайгаар ашиглагдаж байгааг илтгэнэ. Random Forest загвар параметрийн
өөрчлөлтөд харьцангуй тогтвортой байсан бол AdaBoost загвар бүх
тохиргоон дээр ижил гүйцэтгэл үзүүлсэн нь параметрийн мэдрэг байдал бага
байгааг илэрхийлж байна.

6.3 Word2Vec (Skip-gram) + ангилагч загваруудын гүйцэтгэл

| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 10.0       | 0.8557 / 0.8570                | 0.8289 / 0.8323          | 0.8111 / 0.8114     | 0.8403 / 0.8404 |
| 5.0        | 0.8554 / 0.8567                | 0.8281 / 0.8315          | 0.8111 / 0.8114     | 0.8394 / 0.8437 |
| 2.0        | 0.8543 / 0.8556                | 0.8295 / 0.8329          | 0.8111 / 0.8114     | 0.8421 / 0.8428 |
| 1.0        | 0.8554 / 0.8567                | 0.8236 / 0.8274          | 0.8111 / 0.8114     | 0.8406 / 0.8401 |
| 0.5        | 0.8538 / 0.8551                | 0.8293 / 0.8325          | 0.8111 / 0.8114     | 0.8398 / 0.8345 |
| 0.1        | 0.8487 / 0.8503                | 0.8261 / 0.8291          | 0.8111 / 0.8114     | 0.8415 / 0.8451 |
| 0.05       | 0.8423 / 0.8440                | 0.8283 / 0.8314          | 0.8111 / 0.8114     | 0.8398 / 0.8463 |
| 0.01       | 0.8129 / 0.8153                | 0.8261 / 0.8295          | 0.8111 / 0.8114     | 0.8408 / 0.8389 |


Word2Vec-ийн Skip-gram embedding дээр загваруудын гүйцэтгэлийг
параметрийн өөрчлөлт бүрийн хүрээнд Accuracy болон F1-score ашиглан
үнэлэв. Skip-gram нь ховор болон сэтгэл хөдлөл илэрхийлсэн үгсийн
семантик мэдээллийг илүү сайн сурч чаддаг тул Logistic Regression болон
LSTM загваруудтай хослох үед хамгийн өндөр гүйцэтгэлийг үзүүлж байна.
Logistic Regression загварын хувьд C = 10.0 орчим утгад хамгийн сайн үр
дүн гарсан бол LSTM загвар C = 0.05 болон 5.0 утгууд дээр өндөр F1-score
үзүүлсэн нь Skip-gram embedding-ийн dense, семантик төлөөлөл дараалалд
суурилсан загварт тохиромжтой болохыг харуулж байна. Random Forest
загвар параметрийн өөрчлөлтөд харьцангуй тогтвортой байсан бол AdaBoost
загвар бүх тохиргоон дээр ижил гүйцэтгэл үзүүлсэн нь параметрийн мэдрэг
байдал бага байгааг илтгэнэ.

6.4 Хүснэгт: ****BERT (base) + ангилагч загваруудын гүйцэтгэл****

 | C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 1.0        | 0.7849 / 0.7860                | 0.7163 / 0.7138          | 0.7250 / 0.7239     | 0.7645 / 0.7663 |
| 0.5        | 0.7849 / 0.7858                | 0.7177 / 0.7149          | 0.7250 / 0.7239     | 0.7563 / 0.7367 |
| 2.0        | 0.7847 / 0.7859                | 0.7176 / 0.7146          | 0.7250 / 0.7239     | 0.7616 / 0.7547 |
| 5.0        | 0.7835 / 0.7848                | 0.7204 / 0.7178          | 0.7250 / 0.7239     | 0.7542 / 0.7358 |
| 10.0       | 0.7832 / 0.7845                | 0.7188 / 0.7154          | 0.7250 / 0.7239     | 0.7642 / 0.7648 |
| 0.1        | 0.7845 / 0.7848                | 0.7074 / 0.7037          | 0.7250 / 0.7239     | 0.7563 / 0.7431 |
| 0.05       | 0.7824 / 0.7826                | 0.7173 / 0.7142          | 0.7250 / 0.7239     | 0.7618 / 0.7714 |
| 0.01       | 0.7703 / 0.7696                | 0.7191 / 0.7160          | 0.7250 / 0.7239     | 0.7551 / 0.7756 |


BERT (base) embedding дээр загваруудын гүйцэтгэлийг параметрийн өөрчлөлт
бүрийн хүрээнд Accuracy болон F1-score ашиглан үнэлэв. Logistic
Regression загварын хувьд C параметрийн 0.5--1.0 орчим утгуудад хамгийн
тогтвортой, өндөр гүйцэтгэл ажиглагдсан нь BERT-ийн CLS embedding нь
шугаман ангилагчтай сайн зохицож байгааг харуулж байна. LSTM загварын
хувьд зарим параметрийн тохиргоон дээр F1-score илүү өндөр гарсан нь
BERT-ийн dense, контекстэд суурилсан төлөөллийг дараалалд суурилсан
загвар үр ашигтай ашиглаж байгааг илтгэнэ. Харин Random Forest болон
AdaBoost загваруудын гүйцэтгэл BERT embedding дээр харьцангуй бага,
тогтвортой байдалтай байсан нь модонд суурилсан загварууд өндөр
хэмжээст, dense embedding-ийг бүрэн ашиглаж чаддаггүйтэй холбоотой гэж
үзэж болно.

6.5 Хүснэгт: ****SBERT + ангилагч загваруудын гүйцэтгэл****

  | C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 10.0       | 0.8290 / 0.8300                | 0.7824 / 0.7871          | 0.7695 / 0.7704     | 0.8129 / 0.8132 |
| 5.0        | 0.8288 / 0.8298                | 0.7763 / 0.7801          | 0.7695 / 0.7704     | 0.8122 / 0.8108 |
| 2.0        | 0.8274 / 0.8283                | 0.7813 / 0.7862          | 0.7695 / 0.7704     | 0.8057 / 0.8173 |
| 1.0        | 0.8247 / 0.8257                | 0.7852 / 0.7899          | 0.7695 / 0.7704     | 0.8094 / 0.8173 |
| 0.5        | 0.8218 / 0.8228                | 0.7815 / 0.7867          | 0.7695 / 0.7704     | 0.8132 / 0.8160 |
| 0.1        | 0.8143 / 0.8157                | 0.7793 / 0.7846          | 0.7695 / 0.7704     | 0.8109 / 0.8066 |
| 0.05       | 0.8120 / 0.8134                | 0.7827 / 0.7864          | 0.7695 / 0.7704     | 0.8124 / 0.8133 |
| 0.01       | 0.7982 / 0.7994                | 0.7795 / 0.7835          | 0.7695 / 0.7704     | 0.8136 / 0.8186 |


SBERT embedding дээр загваруудын гүйцэтгэлийг параметрийн өөрчлөлт
бүрийн хүрээнд Accuracy болон F1-score ашиглан үнэлэв. Logistic
Regression загвар SBERT embedding-тэй хослох үед C = 5.0--10.0 орчим
утгуудад хамгийн өндөр, тогтвортой гүйцэтгэл үзүүлсэн нь SBERT-ийн
sentence-level семантик төлөөлөл шугаман ангилагчид сайн нийцэж байгааг
харуулж байна. LSTM загварын хувьд C = 0.01--2.0 утгууд дээр харьцангуй
өндөр F1-score ажиглагдсан нь SBERT-ийн contextualized embedding нь
дараалалд суурилсан загварт үр ашигтайгаар ашиглагдаж байгааг илтгэнэ.
Random Forest болон AdaBoost загваруудын гүйцэтгэл харьцангуй бага
байсан нь өндөр хэмжээст, dense embedding-үүд модонд суурилсан
загваруудад тохиромж багатайтай холбоотой гэж үзэж болно.

6.6 Хүснэгт: ****RoBERTa + ангилагч загваруудын гүйцэтгэл****

  | C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 10.0       | 0.8423 / 0.8431                | 0.8071 / 0.8089          | 0.8012 / 0.7981     | 0.8242 / 0.8321 |
| 5.0        | 0.8411 / 0.8418                | 0.7984 / 0.8001          | 0.8012 / 0.7981     | 0.8277 / 0.8293 |
| 2.0        | 0.8385 / 0.8391                | 0.8067 / 0.8091          | 0.8012 / 0.7981     | 0.8264 / 0.8315 |
| 1.0        | 0.8374 / 0.8380                | 0.8054 / 0.8071          | 0.8012 / 0.7981     | 0.8236 / 0.8326 |
| 0.5        | 0.8342 / 0.8348                | 0.8076 / 0.8092          | 0.8012 / 0.7981     | 0.8253 / 0.8322 |
| 0.1        | 0.8290 / 0.8297                | 0.8029 / 0.8044          | 0.8012 / 0.7981     | 0.8253 / 0.8255 |
| 0.05       | 0.8228 / 0.8233                | 0.7997 / 0.8011          | 0.8012 / 0.7981     | 0.8247 / 0.8241 |
| 0.01       | 0.8079 / 0.8083                | 0.7997 / 0.8011          | 0.8012 / 0.7981     | 0.8212 / 0.8119 |


RoBERTa embedding дээр загваруудын гүйцэтгэлийг параметрийн өөрчлөлт
бүрийн хүрээнд Accuracy болон F1-score ашиглан үнэлэв. Logistic
Regression загвар RoBERTa-тай хослох үед хамгийн өндөр, тогтвортой
гүйцэтгэлийг үзүүлсэн бөгөөд ялангуяа C = 10.0 утгад хамгийн өндөр
Accuracy болон F1-score ажиглагдсан. LSTM загварын хувьд C = 1.0 орчим
утгад хамгийн сайн F1-score гарсан нь RoBERTa-гийн контекстэд суурилсан,
баялаг семантик төлөөлөл дараалалд суурилсан загварт үр ашигтайгаар
ашиглагдаж байгааг харуулж байна. Random Forest загварын гүйцэтгэл дунд
зэрэг, параметрийн өөрчлөлтөд харьцангуй тогтвортой байсан бол AdaBoost
загвар бүх тохиргоон дээр бараг ижил гүйцэтгэл үзүүлсэн нь параметрийн
мэдрэг байдал бага байгааг илтгэнэ.

6.7 Хүснэгт: ****HateBERT + ангилагч загваруудын гүйцэтгэл****

| C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 0.05       | 0.8178 / 0.8184                | 0.7605 / 0.7630          | 0.7616 / 0.7601     | 0.8022 / 0.8122 |
| 0.1        | 0.8178 / 0.8185                | 0.7643 / 0.7669          | 0.7616 / 0.7601     | 0.7982 / 0.7858 |
| 0.5        | 0.8191 / 0.8195                | 0.7665 / 0.7689          | 0.7616 / 0.7601     | 0.8014 / 0.8148 |
| 5.0        | 0.8183 / 0.8190                | 0.7640 / 0.7673          | 0.7616 / 0.7601     | 0.8076 / 0.8077 |
| 2.0        | 0.8187 / 0.8192                | 0.7592 / 0.7617          | 0.7616 / 0.7601     | 0.8058 / 0.8087 |
| 10.0       | 0.8184 / 0.8191                | 0.7634 / 0.7663          | 0.7616 / 0.7601     | 0.8097 / 0.8107 |
| 1.0        | 0.8198 / 0.8203                | 0.7602 / 0.7640          | 0.7616 / 0.7601     | 0.8047 / 0.8128 |
| 0.01       | 0.8111 / 0.8116                | 0.7612 / 0.7643          | 0.7616 / 0.7601     | 0.8029 / 0.7953 |


HateBERT embedding дээр загваруудын гүйцэтгэлийг параметрийн өөрчлөлт
бүрийн хүрээнд Accuracy болон F1-score ашиглан үнэлэв. Logistic
Regression загварын хувьд C = 0.5--1.0 орчим утгууд дээр хамгийн өндөр,
тогтвортой гүйцэтгэл ажиглагдсан нь HateBERT-ийн сөрөг, хүчтэй сэтгэл
хөдлөлийг илэрхийлсэн семантик төлөөлөл шугаман ангилагчтай сайн нийцэж
байгааг харуулж байна. LSTM загварын хувьд хэд хэдэн параметрийн
тохиргоон дээр өндөр F1-score гарсан нь HateBERT embedding нь дараалалд
суурилсан загварт илүү баялаг контекстийн мэдээлэл өгч байгааг илтгэнэ.
Харин Random Forest болон AdaBoost загваруудын гүйцэтгэл харьцангуй
бага, тогтвортой байсан нь өндөр хэмжээст, dense embedding-үүд модонд
суурилсан загваруудад тохиромж багатайтай холбоотой гэж үзэж болно.

6.8 ****ALBERT + ангилагч загваруудын гүйцэтгэл****

 | C параметр | Logistic Regression (Acc / F1) | Random Forest (Acc / F1) | AdaBoost (Acc / F1) | LSTM (Acc / F1) |
| ---------- | ------------------------------ | ------------------------ | ------------------- | --------------- |
| 0.5        | 0.8080 / 0.8075                | 0.7409 / 0.7402          | 0.7376 / 0.7315     | 0.7845 / 0.7913 |
| 1.0        | 0.8075 / 0.8069                | 0.7382 / 0.7388          | 0.7376 / 0.7315     | 0.7869 / 0.7839 |
| 5.0        | 0.8072 / 0.8067                | 0.7367 / 0.7360          | 0.7376 / 0.7315     | 0.7804 / 0.7658 |
| 10.0       | 0.8069 / 0.8065                | 0.7410 / 0.7399          | 0.7376 / 0.7315     | 0.7612 / 0.7236 |
| 2.0        | 0.8070 / 0.8065                | 0.7412 / 0.7403          | 0.7376 / 0.7315     | 0.7652 / 0.7908 |
| 0.1        | 0.8099 / 0.8092                | 0.7408 / 0.7404          | 0.7376 / 0.7315     | 0.7842 / 0.7944 |
| 0.05       | 0.8104 / 0.8094                | 0.7356 / 0.7362          | 0.7376 / 0.7315     | 0.7737 / 0.7563 |
| 0.01       | 0.8094 / 0.8087                | 0.7405 / 0.7390          | 0.7376 / 0.7315     | 0.7811 / 0.7927 |


ALBERT embedding ашигласан туршилтуудаас харахад Logistic Regression
загвар нь бага хэмжээний C параметрүүд (ялангуяа 0.05--0.1) дээр хамгийн
тогтвортой, өндөр гүйцэтгэлийг үзүүлсэн нь ALBERT-ийн parameter
sharing-д суурилсан нягт семантик төлөөлөл шугаман ангилагчтай сайн
зохицож байгааг илтгэнэ. LSTM загварын хувьд хэд хэдэн тохиргоон дээр
өндөр F1-score гарсан нь ALBERT embedding нь дарааллын мэдээллийг
тодорхой хэмжээнд хадгалж чаддагийн нотолгоо юм. Харин Random Forest
болон AdaBoost загваруудын гүйцэтгэл харьцангуй тогтвортой боловч
доогуур байсан нь өндөр хэмжээст, dense embedding-үүд модонд суурилсан
ансамбль аргуудад хязгаарлагдмал тохирдогтой холбоотой байж болох юм.

7.ДҮГНЭЛТ

7.1 ЕРӨНХИЙ ДҮГНЭЛТ

Энэхүү судалгаанд sentiment analysis зорилгоор олон төрлийн embedding
арга болон ангиллын загваруудыг ашиглан харьцуулсан туршилтуудыг хийсэн.
Туршилтуудад TF-IDF, Word2Vec (CBOW болон Skip-gram), мөн transformer
суурьтай BERT, SBERT, ALBERT, RoBERTa, HateBERT embedding-үүдийг
ашиглаж, эдгээрийг Logistic Regression, Random Forest, AdaBoost болон
LSTM загваруудтай хослуулан үнэлсэн. Загвар бүр дээр параметрүүдийг олон
утгаар сольж, cross-validation ашиглан гүйцэтгэлийг харьцуулсан.

Ерөнхий дүнгээс харахад уламжлалт TF-IDF embedding нь Logistic
Regression загвартай хосолсон үед тогтвортой, харьцангуй сайн үр дүн
үзүүлсэн. Энэ нь TF-IDF нь текстийн давтамжид суурилсан энгийн боловч
шугаман загваруудад тохиромжтой embedding гэдгийг баталж байна. Харин
Random Forest болон AdaBoost загварууд TF-IDF-ийн өндөр хэмжээст, сийрэг
(sparse) өгөгдөл дээр харьцангуй сул гүйцэтгэл үзүүлсэн нь эдгээр
загварууд ийм төрлийн feature-д төдийлөн тохиромжгүйтэй холбоотой гэж
үзэж байна.

Word2Vec embedding-ийн хувьд CBOW болон Skip-gram аргуудыг харьцуулахад,
хоёр арга хоёулаа TF-IDF-ээс бага зэрэг сул гүйцэтгэлтэй байсан боловч
LSTM загвартай хосолсон үед илүү тогтвортой үр дүн гарсан. Энэ нь
Word2Vec embedding нь үгийн дараалал болон семантик мэдээллийг агуулдаг
тул дараалалд суурилсан deep learning загварт илүү тохиромжтой болохыг
харуулж байна. Гэсэн хэдий ч parameter-ийн өөрчлөлтөөс хамааран үр дүн
их хэлбэлзэж байсан нь Word2Vec загваруудын hyperparameter тохиргоо
чухал болохыг илтгэнэ.

Transformer суурьтай embedding-үүдийн дундаас BERT, RoBERTa болон
HateBERT нь хамгийн өндөр гүйцэтгэлтэй байсан. Ялангуяа эдгээр
embedding-үүдийг Logistic Regression болон LSTM загвартай хослуулах үед
accuracy болон F1-score хамгийн өндөр утгуудад хүрсэн. Энэ нь contextual
embedding нь текстийн утга, өгүүлбэрийн контекстыг илүү сайн барьж
чаддагтай холбоотой гэж үзэж байна. ALBERT болон SBERT нь параметрийн
тоо цөөн, хөнгөн загварууд боловч BERT-тэй ойролцоо гүйцэтгэл үзүүлсэн
нь тооцооллын нөөц багатай орчинд ашиглахад давуу талтайг харуулсан.

Загваруудын хувьд Logistic Regression ихэнх embedding дээр тогтвортой,
найдвартай үр дүн үзүүлсэн бөгөөд параметрийн өөрчлөлтөд харьцангуй бага
мэдрэмтгий байсан. LSTM загвар нь Word2Vec болон transformer
embedding-үүд дээр илүү сайн ажилласан боловч сургалтын хугацаа болон
тооцооллын зардал өндөр байсан. Харин Random Forest болон AdaBoost
загварууд нь өндөр хэмжээст embedding дээр ихэнх тохиолдолд бусад
загваруудаас сул гүйцэтгэл үзүүлсэн.

Дүгнэж хэлбэл, энгийн бөгөөд хурдан шийдэл шаардсан тохиолдолд TF-IDF +
Logistic Regression хослол тохиромжтой бол, хамгийн өндөр гүйцэтгэл
шаардсан үед transformer суурьтай embedding (ялангуяа BERT төрлийн
загварууд) болон Logistic Regression эсвэл LSTM ашиглах нь илүү үр
дүнтэй байна. Судалгааны үр дүнгээс харахад embedding сонголт нь
загварын гүйцэтгэлд хамгийн их нөлөөтэй хүчин зүйл байсан бөгөөд
параметрийн зөв тохиргоо нь загварын үр дүнг мэдэгдэхүйц сайжруулж
чаддаг болох нь харагдсан.

7.2 СУДАЛГААНЫ ХЯЗГААРЛАЛТ 

Энэхүү судалгаанд зарим хязгаарлалтууд ажиглагдсан. Нэгдүгээрт,
туршилтууд нь зөвхөн IMDB sentiment өгөгдлийн сан дээр хийгдсэн тул
гарсан үр дүнг бусад төрлийн текст өгөгдөлд (жишээ нь сошиал сүлжээ,
мэдээ, форум) шууд ерөнхийлөн хэрэглэхэд хүндрэлтэй байж магадгүй.
Өгөгдлийн төрөл өөрчлөгдөх үед embedding болон загваруудын гүйцэтгэл өөр
байж болох юм.

Хоёрдугаарт, BERT, ALBERT, RoBERTa зэрэг transformer суурьтай
embedding-үүдийг зөвхөн embedding гаргах зорилгоор ашигласан бөгөөд
fine-tuning хийгдээгүй. Fine-tuning хийсэн тохиолдолд илүү сайн үр дүн
гарах боломжтой боловч тооцооллын нөөц болон runtime-ийг багасгах
зорилгоор энэ судалгаанд уг аргыг ашиглаагүй.

Гуравдугаарт, параметрийн хайлтыг хязгаарлагдмал хүрээнд (10 өөр
параметрийн утга) хийсэн тул тухайн загваруудын хамгийн оновчтой
параметр бүрэн олдсон эсэх нь баталгаатай биш юм. Мөн cross-validation
нь стандарт k-fold хэлбэрээр хийгдсэн бөгөөд илүү нарийвчилсан
hyperparameter optimization аргууд ашиглаагүй.

Мөн Random Forest, AdaBoost зэрэг загварууд нь өндөр хэмжээст embedding
дээр төдийлөн тохиромжтой биш байж болох бөгөөд энэ нь тэдгээр
загваруудын гүйцэтгэл харьцангуй доогуур гарсан нэг шалтгаан болсон гэж
үзэж байна.

## []{#anchor-17}7.3 ЦААШДЫН СУДАЛГААНЫ ЧИГЛЭЛ

Цаашдын судалгаанд энэхүү ажлыг хэд хэдэн чиглэлээр өргөжүүлэх
боломжтой. Нэгдүгээрт, transformer суурьтай embedding-үүд дээр
fine-tuning хийж, одоогийн feature-based аргын үр дүнтэй харьцуулах нь
илүү гүнзгий дүгнэлт хийх боломж олгоно.

Хоёрдугаарт, өөр төрлийн өгөгдлийн сан ашиглан (жишээлбэл Twitter эсвэл
news sentiment dataset) загваруудын ерөнхийшүүлэх чадварыг шалгах нь
судалгааны ач холбогдлыг нэмэгдүүлнэ. Ингэснээр тухайн embedding болон
загварын бодит хэрэглээнд тохирох эсэхийг илүү сайн үнэлэх боломжтой.

Гуравдугаарт, Word2Vec, TF-IDF зэрэг уламжлалт embedding-үүдийг deep
learning загваруудтай (CNN, attention-based LSTM гэх мэт) хослуулах,
эсвэл hybrid архитектур ашиглах нь гүйцэтгэлийг сайжруулах боломжтой гэж
үзэж байна.

Цаашид мөн загваруудын зөвхөн нарийвчлал бус, сургалтын хугацаа болон
inference time-ийг хамтад нь харьцуулах нь бодит системд ашиглах
боломжийг илүү бодитоор үнэлэхэд чухал ач холбогдолтой юм.

8. ЭХ СУРВАЛЖ

-   IMDB Movie Review Dataset.\
    <https://ai.stanford.edu/~amaas/data/sentiment/>

```{=html}
<!-- -->
```
-   Salton, G., Wong, A., & Yang, C. S. (1975).\
    *A Vector Space Model for Automatic Indexing*.\
    <https://dl.acm.org/doi/10.1145/361219.361220>
-   Semisupervised Autoencoder for Sentiment Analysis\
    https://ojs.aaai.org/index.php/AAAI/article/view/10483
-   Text Based Sentiment Analysis using LSTM\
    <https://www.ijert.org/text-based-sentiment-analysis-using-lstm>
-   Sentiment Classification of Movie Review using Machine Learning
    Approach\
    <https://www.irjet.net/archives/V9/i6/IRJET-V9I6129.pdf>
-   Addressing Sentiment Analysis Challenges within AI Media Platform:
    The Enabling Role of an AI-Powered Chatbot\
    <https://www.mdpi.com/2076-3417/11/4/1685>
-   Adaptive Rider Feedback Artificial Tree Optimization-Based Deep
    Neuro-Fuzzy Network for Classification of Sentiment Grade\
    https://doi.org/10.26636/jtit.2023.164022
-   Baselines and Bigrams: Simple, Good Sentiment and Topic
    Classification\
    <https://aclanthology.org/P12-2018/>
-   Domain Adaptable Model for Sentiment Analysis\
    https://www.sciencedirect.com/science/article/pii/S2452414X22000063
-   Distributed Representations of Sentences and Documents\
    <https://proceedings.mlr.press/v32/le14.html>
-   Sentiment Analysis for Movie Reviews using Artificial Neural
    Networks and Recurrent Neural Networks (Thesis)\
    <https://ir.lib.uwo.ca/etd/8934/>
-   Analysis of IMDB Reviews for Movies and Television Series using SAS®
    Enterprise Miner™ and SAS® Sentiment Analysis Studio\
    https://support.sas.com/resources/papers/proceedings16/SAS1923-2016.pdf
