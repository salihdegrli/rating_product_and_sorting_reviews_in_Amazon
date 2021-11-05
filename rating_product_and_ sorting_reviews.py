from Helper import eda
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def get_data():
    amazon = eda.load_data("amazon_review.csv")
    df = amazon.copy()
    eda.check_df(df)
    df.drop(inplace=True, axis=0, columns="unixReviewTime")
    # I deleted the unixReviewTime column because ı will use reviewTime variable I don't need to unixReviewTime
    return df


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    q1, q2, q3 = dataframe["day_diff"].quantile([0.25, 0.50, 0.75])[:]
    return dataframe.loc[dataframe["day_diff"] <= q1, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q1) & (dataframe["day_diff"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q2) & (dataframe["day_diff"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > q3), "overall"].mean() * w4 / 100


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def main():
    df = get_data()
    print("Görev 1:Average Rating’i güncel yorumlara göre hesaplayınız ve var olan average rating ile kıyaslayınız.\n")

    duty1 = pd.DataFrame([df["overall"].mean(), time_based_weighted_average(df)],
                         index=["avg", "weighted_avg"],
                         columns=["values"]).T
    print(duty1, "\n")

    print("Görev2: Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.\n")
    df['helpful_no'] = df['total_vote'] - df['helpful_yes']
    df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x['helpful_no']),
                                        axis=1)
    print(df.sort_values("wilson_lower_bound",
                         ascending=False)["reviewText"].head(20))


if __name__ == '__main__':
    main()
