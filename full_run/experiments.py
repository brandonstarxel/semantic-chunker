from chroma_research import BaseChunker, GeneralBenchmark
# from chroma_research.chunking import ClusterSemanticChunker
from chroma_research.chunking import ClusterSemanticChunker, LLMSemanticChunker
from chromadb.utils import embedding_functions
from utils import count_non_pad_tokens, num_tokens_from_string
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

import os
OPENAI_API_KEY = os.getenv('OPENAI_CHROMA_API_KEY')

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# Define a custom chunking class
class GregChunker(BaseChunker):
    def split_text(self, text):
        text_splitter = SemanticChunker(OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large"))
        # Custom chunking logic
        return text_splitter.split_text(text)

# Instantiate the custom chunker and benchmark
# chunker = CustomChunker()

# # Choose embedding function
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction()
# default_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = OPENAI_API_KEY, model_name="text-embedding-3-large")
# chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400, length_function=num_tokens_from_string)

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

# class GPTTextChunker(BaseChunker):
#     def __init__(self):
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=50,
#             chunk_overlap=0,
#             length_function=num_tokens_from_string
#             )

#     def get_prompt(self, chunked_input):
#         messages = [
#             {
#                 "role": "system", 
#                 "content": (
#                     "You are an assistant specialized in splitting text into thematically consistent sections. "
#                     "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
#                     "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
#                     "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. "
#                     "Your response should be in the form: 'split_after: 3, 5'."
#                 )
#             },
#             {
#                 "role": "user", 
#                 "content": (
#                     "CHUNKED_TEXT: " + chunked_input + "\n\n"
#                     "Respond only with the IDs of the chunks where you believe a split should occur."
#                 )
#             },
#         ]
#         return messages

#     def split_text(self, text):
#         import re

#         chunks = self.splitter.split_text(text)

#         current_chunk = 0

#         split_indices = []

#         # while True:
#         #     if current_chunk >= len(chunks) - 4:
#         #         break

#         #     token_count = 0

#         #     chunked_input = ''

#         #     for i in range(current_chunk, len(chunks)):
#         #         token_count += num_tokens_from_string(chunks[i])
#         #         chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
#         #         if token_count > 800:
#         #             break

#         #     messages = self.get_prompt(chunked_input)

#         #     completion = self.client.chat.completions.create(
#         #         model="gpt-4o",
#         #         messages=messages
#         #     )

#         #     result_string = completion.choices[0].message.content

#         #     # Use regular expression to find all numbers in the string
#         #     numbers = re.findall(r'\d+', result_string)

#         #     # Convert the found numbers to integers
#         #     numbers = list(map(int, numbers))

#         #     print(numbers)

#         #     split_indices.extend(numbers)

#         #     current_chunk = numbers[-1]

#         #     if len(numbers) == 0:
#         #         break

#         # print(split_indices)

#         if "Good evening. Good evening. If I were smart, I’d go home now." in text:
#             split_indices = [2, 4, 8, 11, 18, 22, 24, 26, 30, 34, 41, 43, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 70, 72, 74, 77, 80, 87, 89, 92, 96, 109, 112, 114, 118, 128, 136, 138, 141, 148, 152, 159, 161, 163, 165, 169, 171, 174, 178, 182, 185, 186, 191, 194, 197, 200, 204, 207, 209, 218, 226, 228, 230, 235, 242, 247, 249, 251, 254, 256, 260, 262, 265, 269, 271, 273, 276, 278, 280, 284, 287, 289, 291, 294, 296, 298, 300, 302, 304, 306, 308]
#         elif " = Valkyria Chronicles III =" in text:
#             split_indices = [2, 4, 7, 9, 12, 16, 18, 20, 22, 25, 27, 30, 34, 36, 38, 40, 42, 44, 50, 52, 54, 56, 60, 63, 67, 70, 72, 78, 89, 91, 94, 97, 100, 103, 106, 108, 110, 112, 116, 118, 121, 124, 126, 128, 131, 134, 137, 140, 142, 146, 152, 159, 161, 164, 168, 172, 176, 178, 180, 184, 186, 189, 192, 194, 196, 198, 200, 203, 206, 212, 214, 216, 219, 221, 223, 226, 228, 230, 231, 233, 236, 239, 242, 244, 246, 248, 251, 258, 266, 270, 272, 275, 279, 283, 288, 290, 292, 294, 297, 302, 304, 306, 308, 310, 312, 314, 317, 320, 321, 324, 326, 329, 331, 333, 335, 346, 349, 350, 352, 356, 358, 364, 366, 368, 370, 373, 376, 380, 384, 386, 388, 390, 392, 394, 397, 399, 401, 403, 405, 407, 408, 411, 414, 418, 422, 424, 427, 430, 433, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 457, 459, 461, 463, 466, 468, 470, 472, 474, 476, 478, 480, 482, 485, 489, 494, 496, 500, 505, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 536, 540, 546, 552, 565, 567, 571, 575, 577, 579, 581, 584, 586, 588, 591, 595, 597, 598, 602, 608, 610, 613, 616, 619, 621, 628, 635, 638, 643, 648, 650, 653, 656, 659]
#         elif "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+)" in text:
#             split_indices = [2, 4, 7, 11, 14, 18, 20, 22, 26, 30, 34, 38, 40, 42, 46, 48, 50, 54, 58, 60, 62, 65, 70, 73, 78, 83, 85, 90, 96, 101, 104, 106, 108, 110, 114, 116, 119, 123, 126, 128, 130, 132, 136, 139, 141, 143, 147, 150, 152, 154, 159, 162, 164, 169, 173, 175, 176, 180, 184, 186, 187, 191, 194, 196, 197, 199, 201, 204, 206, 208, 219, 221, 223, 226, 227, 228, 231, 234, 236, 239, 240, 241, 242, 244, 250, 258, 265, 267, 273, 281, 285, 287, 289, 292, 297, 301, 304, 308]
#         elif "as of december 31, 2017, the company had gross state income tax credit carry-forwards of approximately $20 million, which expire from 2018 through 2020. a deferred tax asset of approximately $16 million (net of federal benefit) has been established related to these state income tax credit carry-forwards, with a valuation allowance of $7 million against such deferred tax asset as of december 31, 2017. the company h" in text:
#             split_indices = [2, 3, 4, 8, 11, 13, 16, 18, 19, 21, 24, 27, 30, 34, 37, 39, 41, 46, 53, 55, 58, 61, 64, 68, 71, 73, 76, 80, 84, 88, 90, 92, 94, 96, 98, 102, 104, 106, 108, 116, 123, 126, 131, 137, 140, 141, 143, 146, 148, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 175, 179, 183, 188, 190, 192, 194, 197, 200, 202, 204, 206, 211, 219, 228, 231, 238, 240, 242, 245, 248, 251, 258, 261, 266, 268, 270, 272, 275, 278, 281, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 305, 317, 320, 329, 335, 337, 341, 344, 351, 355, 358, 362, 371, 373, 375, 378, 382, 386, 390, 392, 396, 399, 402, 405, 407, 409, 411, 413, 416, 418, 425, 427, 430, 439, 446, 448, 451, 454, 457, 460, 462, 466, 470, 476, 478, 480, 482, 485, 491, 496, 510, 512, 514, 518, 522, 526, 528, 530, 532, 537, 540, 542, 546, 554, 558, 560, 562, 564, 566, 570, 572, 578, 580, 586, 595, 597, 598, 601, 603, 606, 609, 615, 619, 626, 628, 630, 633, 636, 641, 644, 646, 648, 651, 655, 662, 664, 666, 670, 676, 678, 681, 684, 692, 694, 696, 698, 702, 704, 708, 710, 712, 714, 716, 718, 720, 722, 724, 726, 727, 730, 732, 734, 738, 742, 744, 748, 755, 761, 763, 765, 767, 770, 773, 776, 778, 781, 786, 793, 795, 797, 803, 809, 812, 815, 818, 824, 830, 836, 841, 849, 851, 854, 857, 860, 863, 870, 879, 883, 889, 894, 896, 898, 901, 904, 908, 911, 915, 919, 921, 924, 927, 931, 934, 938, 939, 941, 947, 953, 959, 962, 966, 969, 971, 972, 974, 976, 978, 980, 986, 991, 994, 1005, 1007, 1011, 1018, 1020, 1026, 1032, 1038, 1046, 1048, 1050, 1052, 1056, 1060, 1062, 1064, 1069, 1076, 1081, 1094, 1099, 1102, 1106, 1108, 1112, 1116, 1121, 1123, 1125, 1128, 1130, 1134, 1136, 1138, 1144, 1150, 1152, 1156, 1164, 1168, 1171, 1174, 1186, 1189, 1195, 1202, 1204, 1208, 1211, 1216, 1218, 1229, 1231, 1234, 1238, 1240, 1252, 1258, 1260, 1262, 1264, 1266, 1268, 1270, 1274, 1282, 1284, 1286, 1290, 1294, 1297, 1300, 1308, 1311, 1313, 1316, 1319, 1324, 1328, 1330, 1334, 1343, 1345, 1348, 1352, 1356, 1359, 1361, 1363, 1366, 1369, 1372, 1374, 1381, 1386, 1388, 1391, 1394, 1397, 1401, 1408, 1412, 1414, 1416, 1418, 1420, 1422, 1424, 1426, 1428, 1430, 1432, 1439, 1446, 1448, 1456, 1460, 1464, 1474, 1476, 1483, 1488, 1490, 1496, 1503, 1505, 1508, 1515, 1517, 1519, 1524, 1531, 1533, 1536, 1545, 1547, 1555, 1558, 1564, 1569, 1572, 1578, 1586, 1588, 1590, 1596, 1604, 1606, 1609, 1611, 1613, 1616, 1624, 1628, 1630, 1632, 1635, 1638, 1641, 1644, 1651, 1653, 1656, 1661, 1666, 1678, 1680, 1690, 1698, 1706, 1708, 1711, 1714, 1717, 1720, 1723, 1725, 1728, 1733, 1736, 1738, 1741, 1745, 1752, 1754, 1756, 1758, 1760, 1763, 1769, 1771, 1773, 1776, 1778, 1781, 1784, 1787, 1792, 1804, 1808, 1812, 1822, 1824, 1829, 1834, 1838, 1840, 1845, 1849, 1853, 1858, 1863, 1869, 1872, 1878, 1886, 1889, 1895, 1902, 1904, 1906, 1908, 1911, 1914, 1917, 1919, 1925, 1936, 1938, 1940, 1942, 1945, 1947, 1949, 1951, 1954, 1957, 1960, 1963, 1964, 1966, 1969, 1974, 1979, 1982, 1984, 1986, 1989, 1992, 1995, 1998, 2000, 2002, 2004, 2005, 2007, 2009, 2010, 2011, 2012, 2014, 2015, 2016, 2017, 2018, 2020, 2022, 2027, 2031, 2034, 2038, 2040, 2044, 2056, 2058, 2061, 2064, 2067, 2069, 2071, 2078, 2084, 2088, 2090, 2095, 2103, 2108, 2109, 2111, 2114, 2120, 2123, 2126, 2137, 2149, 2156, 2158, 2161, 2163, 2166, 2169, 2171, 2173, 2175, 2180, 2184, 2187, 2189, 2194, 2203, 2205, 2209, 2213, 2218, 2220, 2222, 2224, 2226, 2228, 2230, 2234, 2236, 2240, 2249, 2251, 2254, 2262, 2264, 2268, 2272, 2276, 2282, 2284, 2291, 2298, 2300, 2304, 2312, 2315, 2328, 2339, 2344, 2346, 2349, 2353, 2357, 2368, 2374, 2378, 2394, 2396, 2398, 2401, 2405, 2414, 2416, 2421, 2427, 2430, 2432, 2435, 2438, 2442, 2446, 2448, 2454, 2460, 2462, 2464, 2466, 2473, 2475, 2478, 2481, 2485, 2487, 2490, 2493, 2501, 2504, 2506, 2508, 2511, 2513, 2516, 2520, 2525, 2538, 2540, 2543, 2546, 2550, 2552, 2555, 2564, 2566, 2572, 2574, 2576, 2578, 2580, 2583, 2587, 2591, 2594, 2602, 2606, 2608, 2611, 2614, 2617, 2622, 2624, 2626, 2629, 2632, 2636, 2638, 2642, 2656, 2658, 2664, 2668, 2671, 2674, 2681, 2684, 2690, 2695, 2708, 2710, 2713, 2724, 2726, 2730, 2734, 2738, 2744, 2756, 2758, 2766, 2768, 2770, 2779, 2782, 2785, 2791, 2796, 2798, 2802, 2806, 2810, 2812, 2816, 2820, 2824, 2826, 2831, 2837, 2840, 2842, 2845, 2850, 2856, 2857, 2860, 2863, 2867, 2870, 2876, 2878, 2886, 2892, 2895, 2899, 2904, 2908, 2910, 2914, 2923, 2925, 2928, 2932, 2940, 2943, 2951, 2956, 2958, 2960, 2964, 2966, 2970, 2972, 2974, 2978, 2983, 2987, 2989, 2993, 2997, 3001, 3004, 3006, 3014, 3022, 3024, 3026, 3029, 3032, 3035, 3037, 3042, 3048, 3050, 3056, 3062, 3066, 3068, 3070, 3074, 3077, 3082, 3084, 3088, 3094, 3100, 3102, 3105, 3108, 3112, 3116, 3118, 3120, 3122, 3126, 3131, 3134, 3136, 3140, 3144, 3148, 3152, 3165, 3167, 3175, 3180, 3182, 3187, 3194, 3196, 3202, 3208, 3214, 3224, 3226, 3230, 3238, 3240, 3246, 3255, 3257, 3268, 3271, 3276, 3285, 3288, 3294, 3301, 3303, 3306, 3315, 3318, 3320, 3324, 3330, 3332, 3334, 3336, 3340, 3348, 3358, 3367, 3369, 3374, 3382, 3384, 3390, 3396, 3400, 3402, 3404, 3406, 3410, 3414, 3416, 3419, 3424, 3430, 3432, 3435, 3444, 3446, 3458, 3466, 3468, 3476, 3477, 3480, 3494, 3496, 3498, 3501, 3504, 3507, 3512, 3514, 3519, 3528, 3530, 3534, 3540, 3545, 3547, 3549, 3554, 3562, 3564, 3566, 3570, 3577, 3580, 3582, 3586, 3594, 3598, 3601, 3606, 3614, 3618, 3620, 3622, 3625, 3630, 3633, 3635, 3637, 3640, 3644, 3648, 3651, 3654, 3658, 3662, 3667, 3678, 3684, 3695, 3698, 3704, 3710, 3714, 3718, 3725, 3728, 3730, 3734, 3744, 3748, 3754, 3769, 3771, 3774, 3778, 3785, 3787, 3792, 3796, 3804, 3806, 3808, 3811, 3814, 3818, 3824, 3826, 3829]
#         elif "PLoS BiolPLoS BiolpbioplosbiolPLoS Biology1544-91731545-7885Public Library of Science San Francisco, USA 10.1371/journal.pbio." in text:
#             split_indices = [2, 8, 13, 16, 20, 22, 24, 26, 28, 32, 34, 38, 40, 42, 46, 54, 57, 61, 66, 70, 74, 76, 78, 82, 84, 88, 90, 92, 94, 96, 98, 102, 104, 108, 111, 113, 116, 121, 124, 127, 129, 130, 132, 133, 134, 136, 137, 139, 141, 142, 143, 144, 146, 148, 151, 154, 159, 161, 162, 164, 166, 168, 170, 172, 173, 175, 178, 182, 186, 190, 193, 195, 198, 201, 204, 206, 209, 216, 227, 230, 238, 240, 244, 247, 251, 254, 256, 258, 260, 262, 264, 267, 269, 271, 273, 274, 276, 278, 281, 284, 286, 288, 290, 292, 294, 296, 298, 302, 304, 306, 308, 311, 314, 317, 319, 321, 324, 326, 329, 331, 333, 336, 338, 340, 344, 349, 354, 357, 358, 360, 361, 363, 364, 366, 369, 371, 373, 375, 377, 378, 380, 381, 385, 390, 394, 399, 404, 408, 420, 431, 433, 436, 439, 442, 445, 448, 451, 454, 455, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 485, 498, 500, 502, 504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 527, 530, 532, 534, 536, 545, 546, 547, 548, 552, 554, 556, 560, 566, 567, 571, 574, 584, 586, 589, 592, 596, 598, 601, 607, 610, 614, 618, 620, 622, 624, 627, 630, 632, 633, 635, 636, 638, 640, 642, 644, 646, 648, 650, 652, 656, 658, 660, 662, 664, 666, 667, 670, 672, 678, 684, 686, 688, 691, 694, 697, 698, 701, 702, 705, 708, 711, 714, 718, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 748, 750, 753, 759, 764, 766, 771, 777, 784, 786, 790, 794, 798, 804, 808, 814, 818, 820, 823, 826, 829, 832, 835, 837, 839, 841, 843, 846, 850, 854, 858, 869, 871, 875, 879, 886, 888, 890, 892, 895, 901, 904, 906, 911, 913, 918, 925, 927, 930, 936, 941, 943, 946, 951, 956, 958, 961, 964, 969, 974, 976, 980, 983, 991, 993, 996, 1000, 1004, 1008, 1012, 1016, 1020, 1030, 1040, 1042, 1054, 1064, 1068, 1070, 1073, 1075, 1077, 1081, 1089, 1091, 1094, 1097, 1102, 1105, 1107, 1111, 1116, 1122, 1124, 1126, 1129, 1132, 1135, 1138, 1141, 1143, 1146, 1150, 1154, 1158, 1160, 1162, 1166, 1170, 1174, 1176, 1180, 1191, 1193, 1196, 1198, 1200, 1204, 1206, 1208, 1211, 1214, 1217, 1220, 1223, 1225, 1228, 1232, 1236, 1238, 1240, 1243, 1245, 1249, 1253, 1256, 1268, 1273, 1275, 1278, 1282, 1286, 1291, 1294, 1299, 1300, 1302, 1306, 1309, 1312, 1314, 1316, 1318, 1321, 1326, 1334, 1342, 1344, 1357, 1364, 1368, 1374, 1388, 1390, 1394, 1396, 1398, 1400, 1404, 1406, 1410, 1414, 1419, 1422, 1430, 1432, 1434, 1442, 1446, 1448, 1454, 1468, 1470, 1473, 1476, 1481, 1484, 1498, 1501, 1504, 1508, 1511, 1514, 1518, 1526, 1530, 1532, 1546, 1548, 1551, 1556, 1563, 1565, 1567, 1570, 1573, 1578, 1584, 1586, 1588, 1592, 1604, 1606, 1608, 1611, 1614, 1617, 1621, 1623, 1626, 1633, 1640, 1643, 1657, 1661, 1664, 1671, 1674, 1681, 1684, 1690, 1698, 1701, 1704, 1707, 1710, 1713, 1718, 1726, 1736, 1744, 1754, 1756, 1759, 1764, 1767, 1771, 1774, 1780, 1790, 1792, 1794, 1797, 1801, 1804, 1807, 1811, 1814, 1822, 1826, 1830, 1834, 1841, 1843, 1846, 1850, 1854, 1856, 1858, 1861, 1864, 1867, 1870, 1872, 1874, 1877, 1881, 1884, 1888, 1890, 1894, 1896, 1900, 1904, 1906, 1908, 1911, 1914, 1916, 1919, 1922, 1924, 1930, 1938, 1940, 1943, 1946, 1951, 1954, 1957, 1960, 1964, 1968, 1972, 1974, 1978, 1981, 1986, 1993, 2002, 2006, 2011, 2017, 2025, 2036, 2038, 2041, 2045, 2050, 2056, 2058, 2061, 2065, 2070, 2073, 2076, 2078, 2080, 2082, 2085, 2090, 2094, 2098, 2101, 2104, 2106, 2110, 2114, 2123, 2128, 2130, 2132, 2136, 2140, 2149, 2151, 2154, 2160, 2164, 2166, 2170, 2173, 2182, 2184, 2197, 2204, 2206, 2208, 2211, 2214, 2219, 2222, 2230, 2237, 2239, 2241, 2246, 2250, 2254, 2256, 2259, 2265, 2268, 2271, 2276, 2278, 2285, 2290, 2297, 2303, 2309, 2311, 2314, 2321, 2323, 2326, 2330, 2334, 2338, 2340, 2342, 2345, 2347, 2350, 2354, 2364, 2366, 2372, 2378, 2381, 2384, 2390, 2393, 2396, 2398, 2401, 2404, 2407, 2409, 2411, 2414, 2418, 2423, 2429, 2432, 2446, 2451, 2453, 2456, 2460, 2463, 2466, 2469, 2471, 2474, 2483, 2485, 2490, 2494, 2501, 2504, 2512, 2518, 2523, 2525, 2531, 2540, 2542, 2546, 2550, 2555, 2558, 2561, 2564, 2567, 2570, 2573, 2588, 2590, 2594, 2600, 2606, 2608, 2610, 2614, 2618, 2622, 2626, 2628, 2631, 2634, 2637, 2640, 2644, 2647, 2658, 2665, 2667, 2673, 2684, 2687, 2689, 2690, 2692, 2694, 2695, 2697, 2699, 2702, 2704, 2707, 2716, 2722, 2732, 2734, 2737, 2741, 2746, 2748, 2751, 2755, 2759, 2761, 2764, 2768, 2771, 2773, 2776, 2780, 2784, 2788, 2791, 2796, 2804, 2808, 2814, 2817, 2819, 2824, 2831, 2836, 2841, 2843, 2845, 2848, 2851, 2854, 2857, 2860, 2863, 2866, 2868, 2870, 2876, 2882, 2888, 2891, 2894, 2897, 2900, 2903, 2906, 2911, 2914, 2917, 2921, 2928, 2930, 2936, 2940, 2948, 2950, 2954, 2957, 2965, 2968, 2970, 2974, 2982, 2984, 2987, 2990, 2993, 2996, 2999, 3004, 3016, 3018, 3021, 3024, 3027, 3029]
#         chunks_to_split_after = [i - 1 for i in split_indices]
#         print("Chunks to split after: ", chunks_to_split_after)
#         # print("TEXT: ", text[:200])
#         # raise ValueError("Splitting after chunks")
#         # [1, 6, 12, 19, 28, 34, 44, 49, 57, 61, 63, 67, 71, 75, 82, 89, 95, 103, 108, 113, 116, 120, 123, 141, 144, 148, 154, 158, 164, 167, 173, 184, 194, 196, 200, 207, 217, 225, 227, 233, 248, 251, 273, 277, 283, 289, 296, 299, 301, 303, 305, 309]
#         # [11, 17, 21, 26, 40, 43, 50, 57, 65, 76, 79, 89, 92, 110, 116, 122, 130, 133, 141, 146, 148, 150, 154, 159, 161, 165, 172, 179, 184, 197, 198, 205, 211, 217, 219, 224, 232, 237, 244, 251, 255, 260, 268, 274, 281, 288, 293, 299, 303, 308, 311, 315, 316, 319, 320, 328, 332, 344, 350, 358, 366, 376, 380, 388, 393, 402, 414, 431, 432, 436, 441, 444, 450, 458, 464, 468, 470, 475, 478, 480, 481, 488, 494, 501, 506, 512, 520, 522, 530, 537, 540, 547, 552, 559, 563, 569, 574, 576, 580, 590, 596, 601, 607, 614, 618, 624, 635, 638, 641, 648, 651, 661]
#         # [10, 15, 22, 23, 31, 48, 51, 65, 68, 70, 73, 82, 84, 96, 110, 112, 119, 121, 124, 137, 147, 155]
#         # [4, 11, 19, 26, 28, 39, 42, 43, 56, 57, 66, 71, 75, 78, 88, 89, 96, 97, 102, 106, 115, 121, 129, 133, 134, 140, 142, 147, 155, 163, 169, 173, 175, 179, 183, 185, 188, 191, 196, 198, 200, 204, 214, 219, 227, 228, 240, 241, 247, 248, 257, 259, 266, 273, 274, 277, 282, 286, 289, 292, 295, 299, 304, 310, 313, 314, 326, 329, 343, 350, 354, 357, 362, 364, 367, 370, 372, 375, 379, 385, 393, 394, 397, 401, 403, 409, 412, 414, 424, 437, 438, 456, 464, 465, 472, 480, 482, 490, 498, 499, 501, 508, 512, 516, 519, 520, 525, 527, 530, 540, 542, 549, 556, 569, 570, 579, 589, 595, 599, 605, 608, 609, 621, 625, 626, 632, 637, 638, 639, 645, 653, 654, 655, 660, 665, 666, 669, 678, 682, 695, 696, 703, 712, 713, 716, 720, 725, 730, 732, 733, 739, 741, 756, 760, 770, 771, 779, 782, 787, 790, 793, 798, 804, 808, 812, 821, 823, 829, 838, 839, 847, 857, 864, 871, 878, 888, 889, 891, 906, 908, 909, 924, 926, 939, 940, 946, 947, 965, 967, 968, 983, 991, 1000, 1004, 1007, 1019, 1020, 1030, 1039, 1045, 1047, 1049, 1052, 1062, 1065, 1072, 1079, 1089, 1090, 1092, 1095, 1102, 1107, 1112, 1114, 1120, 1124, 1129, 1141, 1144, 1145, 1150, 1153, 1159, 1164, 1172, 1179, 1180, 1195, 1196, 1199, 1204, 1206, 1209, 1218, 1219, 1237, 1239, 1252, 1263, 1270, 1275, 1276, 1282, 1284, 1289, 1294, 1295, 1310, 1311, 1326, 1327, 1333, 1342, 1343, 1358, 1359, 1362, 1368, 1378, 1381, 1388, 1391, 1392, 1396, 1410, 1412, 1416, 1420, 1424, 1430, 1432, 1433, 1439, 1449, 1459, 1460, 1474, 1478, 1485, 1486, 1494, 1503, 1504, 1518, 1524, 1533, 1546, 1549, 1551, 1555, 1559, 1564, 1569, 1570, 1576, 1586, 1588, 1595, 1600, 1603, 1611, 1624, 1631, 1636, 1640, 1642, 1643, 1650, 1653, 1668, 1670, 1682, 1688, 1689, 1698, 1705, 1707, 1711, 1724, 1732, 1733, 1742, 1752, 1758, 1759, 1771, 1781, 1782, 1791, 1796, 1798, 1811, 1813, 1822, 1827, 1831, 1834, 1844, 1845, 1852, 1853, 1858, 1863, 1871, 1877, 1885, 1888, 1890, 1893, 1896, 1898, 1903, 1911, 1919, 1930, 1934, 1944, 1946, 1949, 1957, 1958, 1960, 1964, 1970, 1972, 1975, 1980, 1988, 1990, 1993, 1999, 2006, 2014, 2017, 2021, 2029, 2035, 2049, 2055, 2056, 2061, 2069, 2076, 2083, 2084, 2086, 2094, 2103, 2104, 2120, 2121, 2136, 2139, 2143, 2149, 2153, 2157, 2158, 2161, 2165, 2167, 2176, 2177, 2188, 2198, 2202, 2205, 2210, 2222, 2228, 2230, 2232, 2242, 2244, 2247, 2248, 2262, 2264, 2266, 2273, 2275, 2284, 2286, 2291, 2293, 2296, 2306, 2307, 2313, 2321, 2328, 2329, 2338, 2347, 2354, 2356, 2367, 2368, 2380, 2384, 2385, 2391, 2396, 2397, 2404, 2412, 2413, 2423, 2426, 2438, 2443, 2452, 2456, 2457, 2464, 2472, 2473, 2474, 2487, 2491, 2493, 2494, 2502, 2517, 2523, 2525, 2528, 2535, 2537, 2538, 2549, 2562, 2571, 2573, 2579, 2583, 2589, 2595, 2596, 2602, 2611, 2612, 2619, 2620, 2624, 2627, 2631, 2633, 2637, 2645, 2652, 2659, 2662, 2663, 2669, 2675, 2682, 2684, 2696, 2700, 2701, 2708, 2714, 2715, 2721, 2724, 2731, 2734, 2735, 2743, 2745, 2754, 2758, 2768, 2770, 2777, 2785, 2788, 2796, 2800, 2814, 2816, 2819, 2821, 2830, 2831, 2840, 2848, 2857, 2859, 2874, 2875, 2885, 2887, 2888, 2900, 2904, 2918, 2922, 2923, 2931, 2938, 2940, 2949, 2950, 2962, 2963, 2973, 2977, 2983, 2986, 2988, 2992, 2993, 3006, 3007, 3020, 3029, 3043, 3044, 3062, 3063, 3065, 3073, 3080, 3082, 3083, 3090, 3093, 3098, 3099, 3114, 3115, 3122, 3130, 3132, 3140, 3151, 3158, 3164, 3165, 3172, 3180, 3185, 3193, 3199, 3201, 3208, 3218, 3228, 3230, 3240, 3242, 3255, 3256, 3264, 3265, 3270, 3273, 3274, 3288, 3294, 3303, 3315, 3316, 3318, 3322, 3326, 3329, 3334, 3339, 3341, 3343, 3347, 3357, 3359, 3371, 3374, 3382, 3387, 3390, 3394, 3396, 3406, 3412, 3415, 3418, 3420, 3424, 3430, 3435, 3436, 3448, 3454, 3456, 3468, 3475, 3476, 3480, 3492, 3495, 3501, 3503, 3512, 3529, 3530, 3536, 3540, 3545, 3547, 3549, 3558, 3563, 3565, 3570, 3574, 3577, 3584, 3586, 3587, 3590, 3591, 3604, 3610, 3612, 3615, 3617, 3625, 3632, 3640, 3649, 3650, 3658, 3666, 3667, 3676, 3682, 3691, 3695, 3701, 3703, 3710, 3715, 3724, 3727, 3744, 3749, 3772, 3785, 3786, 3798, 3799, 3809, 3815, 3827]
        
#         chunks = self.splitter.split_text(text)

#         docs = []
#         current_chunk = ''
#         for i, chunk in enumerate(chunks):
#             current_chunk += chunk + ' '
#             if i in chunks_to_split_after:
#                 docs.append(current_chunk.strip())
#                 current_chunk = ''
#         if current_chunk:
#             docs.append(current_chunk.strip())

#         return docs

# SplitterType | ChunkSize | ChunkOverlap | recall | precision | iou_full | iou |
# RecursiveCharacterTextSplitter & 400 & 200 & 0.828 ± 0.373 & 0.143 ± 0.105 \\
# TokenTextSplitter & 400 & 200 & 0.824 ± 0.367 & 0.087 ± 0.051 \\
# RecursiveCharacterTextSplitter & 400 & 0 & 0.831 ± 0.367 & 0.178 ± 0.131 \\
# TokenTextSplitter & 400 & 0 & 0.856 ± 0.338 & 0.129 ± 0.081 \\
# RecursiveCharacterTextSplitter & 200 & 0 & 0.836 ± 0.355 & \textbf{0.302 ± 0.175} \\
# TokenTextSplitter & 200 & 0 & 0.853 ± 0.326 & 0.217 ± 0.122 \\

chunkers = [
    # RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400, length_function=num_tokens_from_string, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    # TokenTextSplitter(chunk_size=800, chunk_overlap=400, encoding_name="cl100k_base"),
    # RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200, length_function=num_tokens_from_string, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    # TokenTextSplitter(chunk_size=400, chunk_overlap=200, encoding_name="cl100k_base"),
    # RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0, length_function=num_tokens_from_string, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    # TokenTextSplitter(chunk_size=400, chunk_overlap=0, encoding_name="cl100k_base"),
    # RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, length_function=num_tokens_from_string, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # ClusterSemanticChunker(default_ef, max_chunk_size=400, length_function=num_tokens_from_string),
    # ClusterSemanticChunker(default_ef, max_chunk_size=200, length_function=num_tokens_from_string),

    # RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=400, length_function=num_tokens_from_string),

    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),
    # TokenTextSplitter(chunk_size=200, chunk_overlap=0, encoding_name="cl100k_base"),

    RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=125, length_function=count_non_pad_tokens, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    TokenTextSplitter(chunk_size=278, chunk_overlap=139, encoding_name="cl100k_base"),
    RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0, length_function=count_non_pad_tokens, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    TokenTextSplitter(chunk_size=278, chunk_overlap=0, encoding_name="cl100k_base"),
    RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, length_function=count_non_pad_tokens, separators = ["\n\n", "\n", ".", "?", "!", " ", ""]),
    TokenTextSplitter(chunk_size=222, chunk_overlap=0, encoding_name="cl100k_base"),
    ClusterSemanticChunker(default_ef, max_chunk_size=250, length_function=count_non_pad_tokens),
    ClusterSemanticChunker(default_ef, max_chunk_size=200, length_function=count_non_pad_tokens)

    # GPTTextChunker()
    # GregChunker()
    # LLMSemanticChunker(api_key=OPENAI_API_KEY),
]

# recall_means = []
db_to_save_chunks = "/Users/brandon/Desktop/MonteIntelligence/AARB_chunks"
# db_to_save_chunks = None
        # print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & {corpus_name} & {recall_scores:.3f} ± {recall_std:.3f} & {precision_scores:.3f} ± {precision_std:.3f} & {brute_iou_mean:.3f} ± {brute_iou_std:.3f} & {iou_scores:.3f} ± {iou_std:.3f}\\\\")
import numpy as np
# Define a function to run the benchmark and print the results
# def run_benchmark_and_print_results(chunker, benchmark, default_ef):
def run_benchmark_and_print_results(chunker, default_ef, retrieve):
    # Run the benchmark
    benchmark = GeneralBenchmark()
    results = benchmark.run(chunker, default_ef, retrieve=retrieve, db_to_save_chunks=db_to_save_chunks)
    del benchmark
    
    chunker_name = chunker.__class__.__name__ if hasattr(chunker, '__class__') else "N/A"
    chunk_size = chunker._chunk_size if hasattr(chunker, '_chunk_size') else "N/A"

    if chunk_size == 278:
        chunk_size = 250
    elif chunk_size == 222:
        chunk_size = 200

    if chunker.__class__.__name__ == "ClusterSemanticChunker":
        chunk_overlap = 0
    else:
        chunk_overlap = chunker._chunk_overlap if hasattr(chunker, '_chunk_overlap') else "N/A"

    if chunk_overlap == 139:
        chunk_overlap = 125

    score_rows = []
    
    corpora_scores = results['corpora_scores']
    for corpus_name, corpus_scores in corpora_scores.items():
        brute_iou_mean = np.mean(corpus_scores['brute_iou_scores'])
        brute_iou_std = np.std(corpus_scores['brute_iou_scores'])
        iou_scores = np.mean(corpus_scores['iou_scores'])
        iou_std = np.std(corpus_scores['iou_scores'])
        recall_scores = np.mean(corpus_scores['recall_scores'])
        recall_std = np.std(corpus_scores['recall_scores'])
        precision_scores = np.mean(corpus_scores['precision_scores'])
        precision_std = np.std(corpus_scores['precision_scores'])

        print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & {corpus_name} & {recall_scores:.3f} ± {recall_std:.3f} & {precision_scores:.3f} ± {precision_std:.3f} & {brute_iou_mean:.3f} ± {brute_iou_std:.3f} & {iou_scores:.3f} ± {iou_std:.3f}\\\\")
        score_rows.append([chunker_name, chunk_size, chunk_overlap, corpus_name, recall_scores, recall_std, precision_scores, precision_std, brute_iou_mean, brute_iou_std, iou_scores, iou_std, retrieve])

        # print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & {corpus_name} & {corpus_scores['recall_mean']:.3f} ± {corpus_scores['recall_std']:.3f} & {corpus_scores['precision_mean']:.3f} ± {corpus_scores['precision_std']:.3f} & {corpus_scores['iou_full_mean']:.3f} ± {corpus_scores['iou_full_std']:.3f} & {corpus_scores['iou_mean']:.3f} ± {corpus_scores['iou_std']:.3f}\\\\")
    print(f"{chunker_name} & {chunk_size} & {chunk_overlap} & ALL & {results['recall_mean']:.3f} ± {results['recall_std']:.3f} & {results['precision_mean']:.3f} ± {results['precision_std']:.3f} & {results['iou_full_mean']:.3f} ± {results['iou_full_std']:.3f} & {results['iou_mean']:.3f} ± {results['iou_std']:.3f}\\\\")
    # print(f"{results['recall_mean']:.3f}")
    score_rows.append([chunker_name, chunk_size, chunk_overlap, "ALL", results['recall_mean'], results['recall_std'], results['precision_mean'], results['precision_std'], results['iou_full_mean'], results['iou_full_std'], results['iou_mean'], results['iou_std'], retrieve])
    # recall_means.append(results['recall_values'])
    
    return score_rows
    # Print the results
    # print(f"iou_full: {results['iou_full_mean']:.3f} ± {results['iou_full_std']:.3f}")
    # print(f"iou: {results['iou_mean']:.3f} ± {results['iou_std']:.3f}")
    # print(f"recall: {results['recall_mean']:.3f} ± {results['recall_std']:.3f}")
    # print(f"precision: {results['precision_mean']:.3f} ± {results['precision_std']:.3f}")

# Define the chunkers with the configurations shown above
import pandas as pd

results_list = []
retrieves = [-1, 5, 10]
# Run the benchmark and print the results for each chunker
for retrieve in retrieves:
    for chunker in chunkers:
        sub_result = run_benchmark_and_print_results(chunker, default_ef, retrieve)
        results_list.extend(sub_result)


# import numpy as np
# recall_means = np.array(recall_means)





# Convert the list of results into a pandas DataFrame
# results_df = pd.DataFrame(results_list, columns=['Chunker', 'Size', 'Overlap', 'Text', 'Recall Mean', 'Recall Std', 'Precision Mean', 'Precision Std' 'IoU Full Mean', 'IoU Full Std' 'IoU Mean', 'IoU Std'])
results_df = pd.DataFrame(results_list)
# Fucking stupid stuff I hate pandas. They won't let me use columns names for no damn reason >:(




# Sort the DataFrame by the column 3 first and then by index
results_df = results_df.sort_values(by=results_df.columns[3], kind='stable')


# print(results_df)

unique_texts = sorted(results_df[3].unique())
# print(unique_texts)

for corpus in unique_texts:
    subset_df = results_df[results_df[3] == corpus]
    max_four_row_index = subset_df[4].idxmax()
    max_six_row_index = subset_df[6].idxmax()
    max_eight_row_index = subset_df[8].idxmax()
    max_ten_row_index = subset_df[10].idxmax()

    print(r"""
\newpage
\begin{table}[htbp]
    \centering
    \setlength{\tabcolsep}{0pt} % Adjust the column separation
    \renewcommand{\arraystretch}{1.2} % Adjust the row separation
    \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l c c c c c c c}
        \toprule
         \textbf{Chunking} & \textbf{Size} & \textbf{Overlap} & \textbf{Retrieve} & \textbf{Recall} & \textbf{Precision } & \textbf{IoU Full} & \textbf{IoU} \\
        \midrule""")
    no_of_chunkers = len(chunkers)
    subset_df_grouped = [subset_df.iloc[n:n+no_of_chunkers] for n in range(0, len(subset_df), no_of_chunkers)]
    for i, subset in enumerate(subset_df_grouped):
        max_four_row_index = subset[4].idxmax()
        max_six_row_index = subset[6].idxmax()
        max_eight_row_index = subset[8].idxmax()
        max_ten_row_index = subset[10].idxmax()

        for index, row in subset.iterrows():
            four_value = f"{row[4]:.3f} ± {row[5]:.3f}"
            four_text = f"\\textbf{{{four_value}}}" if index == max_four_row_index else four_value
            
            six_value = f"{row[6]:.3f} ± {row[7]:.3f}"
            six_text = f"\\textbf{{{six_value}}}" if index == max_six_row_index else six_value
            
            eight_value = f"{row[8]:.3f} ± {row[9]:.3f}"
            eight_text = f"\\textbf{{{eight_value}}}" if index == max_eight_row_index else eight_value

            ten_value = f"{row[10]:.3f} ± {row[11]:.3f}"
            ten_text = f"\\textbf{{{ten_value}}}" if index == max_ten_row_index else ten_value

            retrieve_test = "Min" if row[12] == -1 else str(int(row[12]))
            
            chunker_name = row[0].replace("RecursiveCharacterTextSplitter", "Recursive").replace("TokenTextSplitter", "TokenText").replace("ClusterSemanticChunker", "Cluster")
            corpus_name = row[3].replace("state_of_the_union", "union")
            print(f"{chunker_name} & {row[1]} & {row[2]} & {retrieve_test} & {four_text} & {six_text} & {eight_text} & {ten_text} \\\\")
        if i != len(subset_df_grouped) - 1:
            print("\\\\")
        # print(four_text, six_text, eight_text)
    # print("\\\\")
    print(r"""
        \bottomrule
    \end{tabular*}
    \caption{Performance of various chunking methods on our benchmark over the """+corpus.replace("state_of_the_union",'"state of the union"')+r""" corpus with the Sentence Transformers "all-MiniLM-L6-v2" embedding model.}
    \label{tab:BERT"""+corpus+r"""}
\end{table}""")

# print(default_ef.__class__.__name__)


# ClusterSemanticChunker Max Chunk Size (400), SentenceTransformerEmbeddingFunction
# {'iou_mean': 0.17999288498496147, 'iou_std': 0.12079297826200039, 'recall_mean': 0.7855157992456513, 'recall_std': 0.39622632650842765}

# ClusterSemanticChunker Max Chunk Size (200), SentenceTransformerEmbeddingFunction
# {'iou_mean': 0.2939571209337351, 'iou_std': 0.16339649280179863, 'recall_mean': 0.7200179117047045, 'recall_std': 0.4210548849676757}

# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
#  'recall_mean': 0.7193555455030595, 'recall_std': 0.4291027882174142}