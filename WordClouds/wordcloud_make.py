import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

def make_wordcloud(text, wordcloud_name):
    # lower max_font_size, change the maximum number of word and lighten the background
    wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    wordcloud_name = "./" + wordcloud_name + ".png"
    wordcloud.to_file(wordcloud_name)


