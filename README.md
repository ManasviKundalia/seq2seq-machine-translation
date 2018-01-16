# seq2seq-machine-translation
Machine Translation using seq2seq networks and Noise Contrastive Estimation

I implemented the Machine Translation example from PyTorch using seq2seq networks.
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Furthermore, I have implemented Noise Contrastive Estimation for this use case. 
http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf

Some results that I got using NCE Loss are :
(> input in french language
 = correct translation
 < program out from trained model)

> je serai ton maitre .
= i m going to be your teacher .
< i m going to your your your . <EOS>

> tu es fort courageux .
= you re very brave .
< you re very brave . <EOS>

> elle parle assez vite .
= she speaks fairly quickly .
< she is really a a . <EOS>

> je travaille dans ma ferme .
= i m a farmer .
< i m in my new . <EOS>

> il est a son cote .
= he s at her side .
< he is addicted . <EOS>

> je ne suis pas occupe .
= i m not busy .
< i m not busy . <EOS>

> il se sent beaucoup mieux .
= he s feeling much better .
< he s very better to . . <EOS>

> c est bien .
= i m glad to hear that .
< he s well . <EOS>

> je suis saoul .
= i m drunk .
< i m drunk . <EOS>

> elle s arreta pour fumer une cigarette .
= she stopped to smoke a cigarette .
< she is looking for a a . . <EOS>

> je vais a l etranger cet ete .
= i am going abroad this summer .
< i m going to to the the <EOS>

> je m en sors .
= i m managing .
< i m getting in . <EOS>

> elle reste en contact avec lui .
= she stays in touch with him .
< she is him with him with him . <EOS>

> j en ai assez de jouer .
= i m tired of playing games .
< i m sick of of it . <EOS>

> c est un gentil garcon .
= he s a sweet guy .
< he s a a man . <EOS>

> nous petit dejeunons .
= we are having breakfast .
< we re completely . <EOS>

> je vais sortir cet apres midi .
= i m going to go out this afternoon .
< i m going to be up up tomorrow . <EOS>

> je suis employe de banque .
= i m a bank clerk .
< i m proud of that . <EOS>

> elle n a pas de charme .
= she is curt .
< she is not at <EOS>

> tu es mignon dans ton genre .
= you re kind of cute .
< you re kind of cute . <EOS>

 
