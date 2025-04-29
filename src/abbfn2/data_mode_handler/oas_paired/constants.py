# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing static data used to encode the various OAS-derived data modes.
"""

VALID_GENE_COUNTS = {
    "h_v_call": {  # Total number of categories 299
        "IGHV3-23": 216157,  # 10.640140111561566 of entries %
        "IGHV4-39": 116555,  # 5.737318387575042 of entries %
        "IGHV3-30": 108140,  # 5.323097339731158 of entries %
        "IGHV4-59": 100257,  # 4.935063528661241 of entries %
        "IGHV3-7": 84581,  # 4.163426078156103 of entries %
        "IGHV3-48": 79994,  # 3.937634997174535 of entries %
        "IGHV1-69": 75729,  # 3.727694085819316 of entries %
        "IGHV5-51": 73017,  # 3.594198247227205 of entries %
        "IGHV4-34": 72894,  # 3.5881436793264565 of entries %
        "IGHV3-21": 70802,  # 3.485166800884459 of entries %
        "IGHV1-18": 61713,  # 3.0377686899096443 of entries %
        "IGHV1-2": 59507,  # 2.929180260730368 of entries %
        "IGHV3-15": 55577,  # 2.7357294326820654 of entries %
        "IGHV3-33": 54174,  # 2.666667979310114 of entries %
        "IGHV4-61": 49908,  # 2.4566778438256205 of entries %
        "IGHV1-46": 46865,  # 2.3068888184436904 of entries %
        "IGHV3-9": 46248,  # 2.2765175306814 of entries %
        "IGHV4-4": 45987,  # 2.2636700329407873 of entries %
        "IGHV3-74": 45094,  # 2.2197128854987684 of entries %
        "IGHV2-5": 42271,  # 2.0807531685572016 of entries %
        "IGHV3-11": 41654,  # 2.0503818807949106 of entries %
        "IGHV4-31": 33404,  # 1.6442828142812982 of entries %
        "IGHV1-3": 31374,  # 1.5443578318543123 of entries %
        "IGHV3-49": 29611,  # 1.4575756919435852 of entries %
        "IGHV4-38-2": 29127,  # 1.4337512133747865 of entries %
        "IGHV3-30-3": 28142,  # 1.385265446039525 of entries %
        "IGHV3-53": 26289,  # 1.294053134494104 of entries %
        "IGHV1-8": 22588,  # 1.1118746320496338 of entries %
        "IGHV6-1": 20201,  # 0.9943766354716952 of entries %
        "IGHV7-4-1": 20193,  # 0.9939828424375001 of entries %
        "IGHV3-66": 16749,  # 0.8244549412165448 of entries %
        "IGHV2-70": 16312,  # 0.8029439967236419 of entries %
        "IGHV5-10-1": 16002,  # 0.787684516648585 of entries %
        "IGHV1-24": 14497,  # 0.7136022020906472 of entries %
        "IGHV2-26": 13069,  # 0.6433101454868365 of entries %
        "IGHV3-73": 11246,  # 0.5535745578196467 of entries %
        "IGHV3-72": 10961,  # 0.5395456809764492 of entries %
        "IGHV3-43": 9038,  # 0.44488768038182175 of entries %
        "IGHV4-30-4": 8865,  # 0.4363719060173545 of entries %
        "IGHV3-64D": 8333,  # 0.41018466924338576 of entries %
        "IGHV3-13": 7578,  # 0.37302045164123093 of entries %
        "IGHV4-30-2": 6067,  # 0.29864279230764684 of entries %
        "IGHV3-64": 5961,  # 0.2934250346045629 of entries %
        "IGHV1-58": 5040,  # 0.2480896115428614 of entries %
        "IGHV3-43D": 4450,  # 0.21904737527097884 of entries %
        "IGHV3-20": 4200,  # 0.20674134295238453 of entries %
        "IGHV5-25": 4118,  # 0.2027049643518856 of entries %
        "IGHV5-34": 3608,  # 0.17760065842195316 of entries %
        "IGHV1-69-2": 3073,  # 0.15126574926016134 of entries %
        "IGHV3-1": 2527,  # 0.12438937467635135 of entries %
        "IGHV9-4": 2168,  # 0.10671791226684992 of entries %
        "IGHV5-7": 2073,  # 0.10204161998578407 of entries %
        "IGHV8-20": 1951,  # 0.09603627621431005 of entries %
        "IGHV2-63": 1640,  # 0.08072757200997871 of entries %
        "IGHV5-29": 1633,  # 0.08038300310505807 of entries %
        "IGHV2-1": 1398,  # 0.06881533272557942 of entries %
        "IGHV5-31": 1280,  # 0.0630068854712029 of entries %
        "IGHV1-28": 1250,  # 0.06153016159297158 of entries %
        "IGHV1-82": 1244,  # 0.061234816817325316 of entries %
        "IGHV3-4": 1214,  # 0.059758092939094 of entries %
        "IGHV11-4": 1196,  # 0.058872058612155206 of entries %
        "IGHV1-43": 1175,  # 0.05783835189739329 of entries %
        "IGHV4-2": 1153,  # 0.056755421053356986 of entries %
        "IGHV1-38": 1030,  # 0.05070085315260858 of entries %
        "IGHV5-54": 970,  # 0.04774740539614595 of entries %
        "IGHV3-2": 932,  # 0.04587688848371961 of entries %
        "IGHV3-6": 921,  # 0.04533542306170146 of entries %
        "IGHV5-62": 904,  # 0.04449861286403705 of entries %
        "IGHV1-26": 893,  # 0.043957147442018896 of entries %
        "IGHV1-9": 891,  # 0.04385869918347014 of entries %
        "IGHV7-7": 858,  # 0.04223430291741569 of entries %
        "IGHV1-36": 811,  # 0.03992076884151996 of entries %
        "IGHV2-65": 800,  # 0.03937930341950181 of entries %
        "IGHV9-3": 795,  # 0.039133182773129924 of entries %
        "IGHV1-61": 783,  # 0.0385424932218374 of entries %
        "IGHV5-17": 772,  # 0.03800102779981925 of entries %
        "IGHV5S23": 761,  # 0.0374595623778011 of entries %
        "IGHV6-6": 755,  # 0.03716421760215483 of entries %
        "IGHV14-4": 700,  # 0.03445689049206409 of entries %
        "IGHV5-20": 686,  # 0.033767752682222804 of entries %
        "IGHV10-5": 681,  # 0.033521632035850915 of entries %
        "IGHV1-6": 671,  # 0.033029390743107144 of entries %
        "IGHV2-64": 638,  # 0.031404994477052695 of entries %
        "IGHV2-9-1": 636,  # 0.03130654621850394 of entries %
        "IGHV1-42": 627,  # 0.030863529055034547 of entries %
        "IGHV5-19": 612,  # 0.030125167115918887 of entries %
        "IGHV2-47": 603,  # 0.029682149952449492 of entries %
        "IGHV1-55": 576,  # 0.028353098462041305 of entries %
        "IGHV8S18": 575,  # 0.02830387433276693 of entries %
        "IGHV2-2": 561,  # 0.027614736522925645 of entries %
        "IGHV1-11": 553,  # 0.027220943488730626 of entries %
        "IGHV7-3": 546,  # 0.026876374583809988 of entries %
        "IGHV1-39": 534,  # 0.02628568503251746 of entries %
        "IGHV2-9": 501,  # 0.02466128876646301 of entries %
        "IGHV6-8": 493,  # 0.024267495732267993 of entries %
        "IGHV4-28": 479,  # 0.02357835792242671 of entries %
        "IGHV1S81": 466,  # 0.022938444241859805 of entries %
        "IGHV2S18": 462,  # 0.022741547724762296 of entries %
        "IGHV5S13": 452,  # 0.022249306432018525 of entries %
        "IGHV1-13": 449,  # 0.022101634044195392 of entries %
        "IGHV1-14": 444,  # 0.021855513397823507 of entries %
        "IGHV1-25": 442,  # 0.02175706513927475 of entries %
        "IGHV8-17": 440,  # 0.021658616880725998 of entries %
        "IGHV1-45": 430,  # 0.021166375587982223 of entries %
        "IGHV1-47": 413,  # 0.02032956539031781 of entries %
        "IGHV1-31": 409,  # 0.0201326688732203 of entries %
        "IGHV5-46": 401,  # 0.019738875839025283 of entries %
        "IGHV8-8": 398,  # 0.01959120345120215 of entries %
        "IGHV1-65": 396,  # 0.019492755192653397 of entries %
        "IGHV6-22": 390,  # 0.019197410417007135 of entries %
        "IGHV9-1": 386,  # 0.019000513899909626 of entries %
        "IGHV5-4": 375,  # 0.018459048477891475 of entries %
        "IGHV1-72": 369,  # 0.01816370370224521 of entries %
        "IGHV1-7": 367,  # 0.018065255443696457 of entries %
        "IGHV8-28": 367,  # 0.018065255443696457 of entries %
        "IGHV5-58": 366,  # 0.01801603131442208 of entries %
        "IGHV14-3": 365,  # 0.0179668071851477 of entries %
        "IGHV9-2-1": 344,  # 0.01693310047038578 of entries %
        "IGHV5-22": 341,  # 0.01678542808256265 of entries %
        "IGHV8-23": 339,  # 0.016686979824013893 of entries %
        "IGHV2-8": 331,  # 0.016293186789818875 of entries %
        "IGHV9-3-1": 313,  # 0.015407152462880084 of entries %
        "IGHV5S14": 310,  # 0.015259480075056953 of entries %
        "IGHV1-57": 309,  # 0.015210255945782574 of entries %
        "IGHV2S30": 300,  # 0.01476723878231318 of entries %
        "IGHV5-35": 298,  # 0.014668790523764425 of entries %
        "IGHV11-3": 296,  # 0.01457034226521567 of entries %
        "IGHV5-9-1": 295,  # 0.014521118135941292 of entries %
        "IGHV5-50": 293,  # 0.01442266987739254 of entries %
        "IGHV8-12": 291,  # 0.014324221618843785 of entries %
        "IGHV2-41": 289,  # 0.01422577336029503 of entries %
        "IGHV3-8": 286,  # 0.014078100972471898 of entries %
        "IGHV2-32": 284,  # 0.013979652713923143 of entries %
        "IGHV5-6": 279,  # 0.013733532067551258 of entries %
        "IGHV1-74": 267,  # 0.01314284251625873 of entries %
        "IGHV2S13": 266,  # 0.013093618386984352 of entries %
        "IGHV2-16": 264,  # 0.012995170128435598 of entries %
        "IGHV1S56": 259,  # 0.012749049482063712 of entries %
        "IGHV6-13": 259,  # 0.012749049482063712 of entries %
        "IGHV1-64": 247,  # 0.012158359930771185 of entries %
        "IGHV1-15": 246,  # 0.012109135801496806 of entries %
        "IGHV2-3": 238,  # 0.01171534276730179 of entries %
        "IGHV1-53": 236,  # 0.011616894508753035 of entries %
        "IGHV1-59": 234,  # 0.01151844625020428 of entries %
        "IGHV2-72": 231,  # 0.011370773862381148 of entries %
        "IGHV5-6-5": 228,  # 0.011223101474558017 of entries %
        "IGHV5S10": 228,  # 0.011223101474558017 of entries %
        "IGHV9-6": 228,  # 0.011223101474558017 of entries %
        "IGHV2-30": 225,  # 0.011075429086734884 of entries %
        "IGHV1-16": 223,  # 0.01097698082818613 of entries %
        "IGHV1-5": 219,  # 0.01078008431108862 of entries %
        "IGHV4-1": 216,  # 0.01063241192326549 of entries %
        "IGHV1-50": 215,  # 0.010583187793991112 of entries %
        "IGHV2-43": 210,  # 0.010337067147619226 of entries %
        "IGHV7-6": 208,  # 0.010238618889070471 of entries %
        "IGHV5-27": 197,  # 0.00969715346705232 of entries %
        "IGHV1-60": 194,  # 0.00954948107922919 of entries %
        "IGHV2-6-7": 193,  # 0.009500256949954813 of entries %
        "IGHV1S135": 192,  # 0.009451032820680435 of entries %
        "IGHV6-14": 190,  # 0.00935258456213168 of entries %
        "IGHV1-87": 180,  # 0.008860343269387908 of entries %
        "IGHV1-54": 177,  # 0.008712670881564777 of entries %
        "IGHV1-19": 177,  # 0.008712670881564777 of entries %
        "IGHV1S29": 176,  # 0.008663446752290398 of entries %
        "IGHV8-22": 173,  # 0.008515774364467267 of entries %
        "IGHV1-63": 172,  # 0.00846655023519289 of entries %
        "IGHV1-35": 172,  # 0.00846655023519289 of entries %
        "IGHV1-12": 171,  # 0.008417326105918513 of entries %
        "IGHV2-52": 170,  # 0.008368101976644135 of entries %
        "IGHV2-77": 167,  # 0.008220429588821004 of entries %
        "IGHV1S137": 166,  # 0.008171205459546626 of entries %
        "IGHV2S63": 165,  # 0.008121981330272249 of entries %
        "IGHV10-1": 164,  # 0.008072757200997871 of entries %
        "IGHV6-3": 164,  # 0.008072757200997871 of entries %
        "IGHV1-75": 164,  # 0.008072757200997871 of entries %
        "IGHV1-80": 161,  # 0.00792508481317474 of entries %
        "IGHV1-76": 159,  # 0.007826636554625985 of entries %
        "IGHV11-2": 158,  # 0.007777412425351608 of entries %
        "IGHV1S22": 155,  # 0.007629740037528476 of entries %
        "IGHV1-4": 151,  # 0.007432843520430967 of entries %
        "IGHV2-61": 151,  # 0.007432843520430967 of entries %
        "IGHV1-32": 147,  # 0.007235947003333458 of entries %
        "IGHV5-12-1": 145,  # 0.0071374987447847035 of entries %
        "IGHV8-31": 145,  # 0.0071374987447847035 of entries %
        "IGHV1-52": 143,  # 0.007039050486235949 of entries %
        "IGHV1-29": 137,  # 0.006743705710589685 of entries %
        "IGHV5-9-4": 134,  # 0.006596033322766553 of entries %
        "IGHV1-81": 132,  # 0.006497585064217799 of entries %
        "IGHV8-16": 128,  # 0.00630068854712029 of entries %
        "IGHV3-3": 125,  # 0.006153016159297158 of entries %
        "IGHV14-1": 123,  # 0.006054567900748403 of entries %
        "IGHV7-1": 122,  # 0.006005343771474027 of entries %
        "IGHV6-5": 122,  # 0.006005343771474027 of entries %
        "IGHV14-2": 122,  # 0.006005343771474027 of entries %
        "IGHV3-5": 121,  # 0.0059561196421996495 of entries %
        "IGHV2-45": 118,  # 0.005808447254376518 of entries %
        "IGHV5-9-3": 115,  # 0.005660774866553386 of entries %
        "IGHV10S3": 115,  # 0.005660774866553386 of entries %
        "IGHV5-6-4": 109,  # 0.005365430090907122 of entries %
        "IGHV10-10": 100,  # 0.004922412927437727 of entries %
    },
    "h_d_call": {  # Total number of categories 59
        "IGHD3-10": 212808,  # 10.475288502621677 of entries %
        "IGHD3-22": 173978,  # 8.563915562897607 of entries %
        "IGHD6-19": 152860,  # 7.524400400881309 of entries %
        "IGHD6-13": 135563,  # 6.672970636822405 of entries %
        "IGHD2-15": 128770,  # 6.33859112666156 of entries %
        "IGHD1-26": 125899,  # 6.197268651514824 of entries %
        "IGHD3-3": 122207,  # 6.015533166233823 of entries %
        "IGHD4-17": 117962,  # 5.806576737464091 of entries %
        "IGHD2-2": 116639,  # 5.74145321443409 of entries %
        "IGHD3-16": 95229,  # 4.687564606669673 of entries %
        "IGHD5-12": 85313,  # 4.199458140784948 of entries %
        "IGHD5-18": 68381,  # 3.365995183911192 of entries %
        "IGHD2-21": 67839,  # 3.3393157058444793 of entries %
        "IGHD1-1": 61717,  # 3.0379655864267416 of entries %
        "IGHD3-9": 59086,  # 2.908456902305855 of entries %
        "IGHD6-6": 47582,  # 2.342182519133419 of entries %
        "IGHD2-8": 38407,  # 1.8905511330410076 of entries %
        "IGHD4-4": 25507,  # 1.255559865401541 of entries %
        "IGHD1-7": 21734,  # 1.0698372256493154 of entries %
        "IGHD7-27": 18921,  # 0.9313697500004923 of entries %
        "IGHD1-20": 13394,  # 0.659307987501009 of entries %
        "IGHD6-25": 10608,  # 0.522169563342594 of entries %
        "IGHD1-11": 6537,  # 0.3217781330666042 of entries %
        "IGHD1-2": 5985,  # 0.2946064137071479 of entries %
        "IGHD1-12": 5936,  # 0.29219443137270346 of entries %
        "IGHD5-1": 4233,  # 0.20836573921843896 of entries %
        "IGHD1-6": 3886,  # 0.19128496636023004 of entries %
        "IGHD1-4": 3572,  # 0.17582858976807558 of entries %
        "IGHD2-4": 3229,  # 0.1589447134269642 of entries %
        "IGHD1-9": 3216,  # 0.15830479974639727 of entries %
        "IGHD4-3": 2654,  # 0.13064083909419727 of entries %
        "IGHD1-3": 2510,  # 0.12355256447868694 of entries %
        "IGHD1-10": 2506,  # 0.12335566796158942 of entries %
        "IGHD4-1": 2465,  # 0.12133747866133995 of entries %
        "IGHD2-1": 2428,  # 0.119516185878188 of entries %
        "IGHD2-3": 1843,  # 0.0907200702526773 of entries %
        "IGHD2-5": 1416,  # 0.06970136705251821 of entries %
        "IGHD2-14": 1326,  # 0.06527119541782425 of entries %
        "IGHD1-5": 1192,  # 0.0586751620950577 of entries %
        "IGHD3-1": 1175,  # 0.05783835189739329 of entries %
        "IGHD2-10": 1015,  # 0.04996249121349292 of entries %
        "IGHD3-2": 625,  # 0.03076508079648579 of entries %
        "IGHD2-12": 608,  # 0.029928270598821378 of entries %
        "IGHD4-2": 480,  # 0.02362758205170109 of entries %
        "IGHD4-6": 401,  # 0.019738875839025283 of entries %
        "IGHD3-8": 309,  # 0.015210255945782574 of entries %
        "IGHD3-4": 244,  # 0.012010687542948054 of entries %
        "IGHD1-8": 183,  # 0.00900801565721104 of entries %
    },
    "h_j_call": {  # Total number of categories 11
        "IGHJ4": 982082,  # 48.342131326038974 of entries %
        "IGHJ6": 419713,  # 20.660006970136706 of entries %
        "IGHJ3": 264362,  # 13.012989263232923 of entries %
        "IGHJ5": 237374,  # 11.68452846237603 of entries %
        "IGHJ2": 91466,  # 4.502334208210191 of entries %
        "IGHJ1": 36522,  # 1.7977636493588065 of entries %
    },
    "l_v_call": {  # Total number of categories 297
        "IGKV3-20": 193921,  # 9.545592373016513 of entries %
        "IGKV1-39": 167881,  # 8.263796046711729 of entries %
        "IGKV4-1": 121703,  # 5.990724205079537 of entries %
        "IGLV2-14": 109929,  # 5.411159307003018 of entries %
        "IGKV3-15": 109179,  # 5.374241210047235 of entries %
        "IGKV1-5": 103504,  # 5.094894276415144 of entries %
        "IGKV3-11": 102030,  # 5.022337909864713 of entries %
        "IGKV2-28": 67522,  # 3.3237116568645018 of entries %
        "IGLV1-40": 66631,  # 3.279852957681032 of entries %
        "IGLV1-51": 63298,  # 3.1157889348095322 of entries %
        "IGLV3-1": 61982,  # 3.0510099806844515 of entries %
        "IGLV1-44": 60024,  # 2.954629135565221 of entries %
        "IGKV1-33": 57387,  # 2.8248251066686882 of entries %
        "IGLV3-21": 53763,  # 2.6464368621783447 of entries %
        "IGLV1-47": 41335,  # 2.034679383556384 of entries %
        "IGLV2-23": 40987,  # 2.017549386568901 of entries %
        "IGLV2-8": 40226,  # 1.9800898241910998 of entries %
        "IGLV3-25": 37789,  # 1.8601306211494424 of entries %
        "IGLV3-19": 37774,  # 1.859392259210327 of entries %
        "IGKV2-30": 33602,  # 1.6540291918776249 of entries %
        "IGLV2-11": 30041,  # 1.4787420675315675 of entries %
        "IGKV1-9": 29172,  # 1.4359662991921336 of entries %
        "IGKV1-12": 26532,  # 1.3060145979077775 of entries %
        "IGKV1-17": 24134,  # 1.187975135907821 of entries %
        "IGKV1-8": 21512,  # 1.0589094689504037 of entries %
        "IGKV1-27": 20769,  # 1.0223359408995414 of entries %
        "IGKV1-16": 17980,  # 0.8850498443533032 of entries %
        "IGLV6-57": 17619,  # 0.867279933685253 of entries %
        "IGLV4-69": 14930,  # 0.7349162500664526 of entries %
        "IGLV3-10": 13805,  # 0.6795391046327781 of entries %
        "IGLV7-46": 13580,  # 0.6684636755460432 of entries %
        "IGKV2-24": 13134,  # 0.646509713889671 of entries %
        "IGLV8-61": 11840,  # 0.5828136906086269 of entries %
        "IGKV1-6": 11102,  # 0.5464862832041364 of entries %
        "IGKV2D-29": 9527,  # 0.4689582795969922 of entries %
        "IGLV7-43": 9228,  # 0.4542402649439534 of entries %
        "IGLV3-9": 8090,  # 0.3982232058297121 of entries %
        "IGKV2-29": 6749,  # 0.3322136484727722 of entries %
        "IGKV1D-12": 6495,  # 0.31971071963708037 of entries %
        "IGKV1-NL1": 6263,  # 0.3082907216454248 of entries %
        "IGKV1D-8": 5937,  # 0.29224365550197784 of entries %
        "IGLV10-54": 5731,  # 0.2821034848714561 of entries %
        "IGLV2-18": 5032,  # 0.2476958185086664 of entries %
        "IGKV6-21": 4840,  # 0.23824478568798596 of entries %
        "IGLV5-45": 4801,  # 0.23632504464628526 of entries %
        "IGLV1-36": 4362,  # 0.21471565189483363 of entries %
        "IGKV2-40": 4173,  # 0.20541229146197634 of entries %
        "IGKV3D-15": 3974,  # 0.19561668973637525 of entries %
        "IGKV1-13": 3450,  # 0.16982324599660156 of entries %
        "IGLV4-60": 3231,  # 0.15904316168551294 of entries %
        "IGLV9-49": 3147,  # 0.15490833482646527 of entries %
        "IGKV3D-20": 2997,  # 0.14752471543530868 of entries %
        "IGLV3-27": 2954,  # 0.14540807787651044 of entries %
        "IGKV1D-16": 2551,  # 0.1255707537789364 of entries %
        "IGKV16S1": 2221,  # 0.1093267911183919 of entries %
        "IGKV3S19": 2108,  # 0.10376446451038727 of entries %
        "IGKV10-96": 1930,  # 0.09500256949954812 of entries %
        "IGKV12S36": 1810,  # 0.08909567398662285 of entries %
        "IGKV12S34": 1758,  # 0.08653601926435524 of entries %
        "IGKV1D-13": 1636,  # 0.0805306754928812 of entries %
        "IGKV6D-21": 1477,  # 0.07270403893825522 of entries %
        "IGKV8-30": 1431,  # 0.07043972899163387 of entries %
        "IGKV22S2": 1428,  # 0.07029205660381073 of entries %
        "IGKV1-117": 1307,  # 0.06433593696161109 of entries %
        "IGKV3S10": 1259,  # 0.061973178756440976 of entries %
        "IGKV22S4": 1238,  # 0.06093947204167906 of entries %
        "IGKV15S2": 1215,  # 0.05980731706836838 of entries %
        "IGKV22S1": 1179,  # 0.05803524841449079 of entries %
        "IGKV12S16": 1164,  # 0.05729688647537514 of entries %
        "IGKV5-2": 1038,  # 0.0510946461868036 of entries %
        "IGLV5-37": 1033,  # 0.05084852554043171 of entries %
        "IGKV6S10": 1031,  # 0.05075007728188296 of entries %
        "IGKV8S5": 996,  # 0.04902723275727976 of entries %
        "IGKV1-110": 989,  # 0.048682663852359115 of entries %
        "IGKV2S27": 952,  # 0.04686137106920716 of entries %
        "IGKV22S7": 939,  # 0.04622145738864025 of entries %
        "IGKV3S9": 918,  # 0.04518775067387833 of entries %
        "IGKV1S14": 909,  # 0.04474473351040893 of entries %
        "IGKV14-111": 905,  # 0.04454783699331143 of entries %
        "IGKV1S22": 894,  # 0.04400637157129327 of entries %
        "IGLV1": 883,  # 0.043464906149275125 of entries %
        "IGKV3S11": 873,  # 0.042972664856531353 of entries %
        "IGKV8S4": 855,  # 0.042086630529592564 of entries %
        "IGKV12-46": 836,  # 0.04115137207337939 of entries %
        "IGKV8S6": 828,  # 0.04075757903918437 of entries %
        "IGKV12S11": 826,  # 0.04065913078063562 of entries %
        "IGKV19S2": 818,  # 0.0402653377464406 of entries %
        "IGKV17-121": 816,  # 0.04016688948789185 of entries %
        "IGKV12S38": 815,  # 0.04011766535861747 of entries %
        "IGKV6S5": 809,  # 0.03982232058297121 of entries %
        "IGKV14S9": 805,  # 0.0396254240658737 of entries %
        "IGKV6-15": 784,  # 0.038591717351111776 of entries %
        "IGKV12S29": 758,  # 0.03731188998997797 of entries %
        "IGKV1-135": 739,  # 0.036376631533764796 of entries %
        "IGKV3-4": 720,  # 0.03544137307755163 of entries %
        "IGKV17-127": 695,  # 0.0342107698456922 of entries %
        "IGKV12S31": 695,  # 0.0342107698456922 of entries %
        "IGKV22S6": 683,  # 0.033620080294399675 of entries %
        "IGKV3S1": 675,  # 0.03322628726020466 of entries %
        "IGKV12S32": 674,  # 0.03317706313093028 of entries %
        "IGKV10-94": 672,  # 0.03307861487238152 of entries %
        "IGKV2S23": 644,  # 0.03170033925269896 of entries %
        "IGKV12-44": 641,  # 0.031552666864875824 of entries %
        "IGKV6S11": 640,  # 0.03150344273560145 of entries %
        "IGKV19-93": 630,  # 0.031011201442857676 of entries %
        "IGKV10S5": 629,  # 0.0309619773135833 of entries %
        "IGKV5S12": 622,  # 0.030617408408662658 of entries %
        "IGKV3S18": 621,  # 0.03056818427938828 of entries %
        "IGKV1S21": 618,  # 0.03042051189156515 of entries %
        "IGKV12S39": 616,  # 0.030322063633016396 of entries %
        "IGKV2S6": 610,  # 0.03002671885737013 of entries %
        "IGKV14S1": 604,  # 0.02973137408172387 of entries %
        "IGKV4-59": 584,  # 0.028746891496236323 of entries %
        "IGKV6-23": 583,  # 0.028697667366961947 of entries %
        "IGKV3-10": 561,  # 0.027614736522925645 of entries %
        "IGKV1S26": 561,  # 0.027614736522925645 of entries %
        "IGKV12S8": 543,  # 0.026728702195986855 of entries %
        "IGKV4S7": 542,  # 0.02667947806671248 of entries %
        "IGKV12S26": 532,  # 0.026187236773968704 of entries %
        "IGKV2S25": 531,  # 0.026138012644694328 of entries %
        "IGKV4S5": 530,  # 0.02608878851541995 of entries %
        "IGKV8-24": 524,  # 0.025793443739773686 of entries %
        "IGKV3-1": 517,  # 0.025448874834853048 of entries %
        "IGKV2-137": 506,  # 0.024907409412834897 of entries %
        "IGKV16-104": 473,  # 0.023283013146780447 of entries %
        "IGKV6-17": 467,  # 0.02298766837113418 of entries %
        "IGKV14S18": 465,  # 0.02288922011258543 of entries %
        "IGKV9-120": 460,  # 0.022643099466213543 of entries %
        "IGKV12S30": 457,  # 0.02249542707839041 of entries %
        "IGKV10S9": 455,  # 0.022396978819841654 of entries %
        "IGKV5-43": 453,  # 0.0222985305612929 of entries %
        "IGKV2S11": 433,  # 0.021314047975805356 of entries %
        "IGKV2S17": 431,  # 0.021215599717256603 of entries %
        "IGKV8S9": 424,  # 0.02087103081233596 of entries %
        "IGLV5-39": 411,  # 0.020231117131769057 of entries %
        "IGKV4-57-1": 411,  # 0.020231117131769057 of entries %
        "IGKV1S23": 402,  # 0.01978809996829966 of entries %
        "IGKV2S26": 393,  # 0.019345082804830264 of entries %
        "IGKV17S1": 384,  # 0.01890206564136087 of entries %
        "IGKV2D-30": 380,  # 0.01870516912426336 of entries %
        "IGKV14S15": 379,  # 0.018655944994988984 of entries %
        "IGKV3-2": 370,  # 0.01821292783151959 of entries %
        "IGKV13-84": 357,  # 0.017573014150952682 of entries %
        "IGKV6S8": 353,  # 0.017376117633855173 of entries %
        "IGKV12-41": 352,  # 0.017326893504580797 of entries %
        "IGKV4S4": 348,  # 0.017129996987483288 of entries %
        "IGKV22S8": 344,  # 0.01693310047038578 of entries %
        "IGKV3S6": 342,  # 0.016834652211837026 of entries %
        "IGKV5S2": 341,  # 0.01678542808256265 of entries %
        "IGKV2S16": 340,  # 0.01673620395328827 of entries %
        "IGKV14S13": 340,  # 0.01673620395328827 of entries %
        "IGKV3D-7": 331,  # 0.016293186789818875 of entries %
        "IGKV3-12": 329,  # 0.016194738531270122 of entries %
        "IGKV8-28": 324,  # 0.015948617884898233 of entries %
        "IGKV4S12": 315,  # 0.015505600721428838 of entries %
        "IGKV12S20": 314,  # 0.015456376592154462 of entries %
        "IGKV5-48": 313,  # 0.015407152462880084 of entries %
        "IGKV13-85": 312,  # 0.015357928333605707 of entries %
        "IGKV8-27": 310,  # 0.015259480075056953 of entries %
        "IGKV4-57": 299,  # 0.014718014653038802 of entries %
        "IGKV4-72": 294,  # 0.014471894006666916 of entries %
        "IGKV6-25": 286,  # 0.014078100972471898 of entries %
        "IGKV3-7": 281,  # 0.013831980326100012 of entries %
        "IGKV3-5": 279,  # 0.013733532067551258 of entries %
        "IGKV14S19": 278,  # 0.01368430793827688 of entries %
        "IGKV3S8": 277,  # 0.013635083809002503 of entries %
        "IGKV4-50": 270,  # 0.013290514904081861 of entries %
        "IGKV1S1": 269,  # 0.013241290774807485 of entries %
        "IGKV6-32": 265,  # 0.013044394257709976 of entries %
        "IGKV3S13": 263,  # 0.012945945999161221 of entries %
        "IGKV1S18": 259,  # 0.012749049482063712 of entries %
        "IGKV12S17": 259,  # 0.012749049482063712 of entries %
        "IGKV5S10": 253,  # 0.012453704706417448 of entries %
        "IGKV12S24": 253,  # 0.012453704706417448 of entries %
        "IGKV4-55": 248,  # 0.012207584060045561 of entries %
        "IGKV4S9": 238,  # 0.01171534276730179 of entries %
        "IGKV12S14": 232,  # 0.011419997991655526 of entries %
        "IGKV6S4": 231,  # 0.011370773862381148 of entries %
        "IGKV3D-11": 227,  # 0.011173877345283639 of entries %
        "IGKV8-21": 226,  # 0.011124653216009263 of entries %
        "IGKV4-68": 224,  # 0.011026204957460508 of entries %
        "IGKV6S9": 220,  # 0.010829308440362999 of entries %
        "IGKV1S28": 218,  # 0.010730860181814244 of entries %
        "IGKV2S20": 218,  # 0.010730860181814244 of entries %
        "IGLV3-22": 213,  # 0.010484739535442357 of entries %
        "IGKV5-39": 211,  # 0.010386291276893602 of entries %
        "IGKV2-109": 208,  # 0.010238618889070471 of entries %
        "IGKV9-124": 205,  # 0.010090946501247339 of entries %
        "IGKV6-20": 204,  # 0.010041722371972962 of entries %
        "IGKV1S30": 198,  # 0.009746377596326699 of entries %
        "IGKV4S18": 197,  # 0.00969715346705232 of entries %
        "IGKV4S21": 195,  # 0.009598705208503568 of entries %
        "IGKV4-53": 194,  # 0.00954948107922919 of entries %
        "IGKV14S8": 194,  # 0.00954948107922919 of entries %
        "IGKV3S5": 184,  # 0.009057239786485417 of entries %
        "IGKV4-91": 180,  # 0.008860343269387908 of entries %
        "IGKV4S2": 178,  # 0.008761895010839153 of entries %
        "IGLV5-52": 177,  # 0.008712670881564777 of entries %
        "IGKV8-19": 177,  # 0.008712670881564777 of entries %
        "IGKV4S13": 174,  # 0.008564998493741644 of entries %
        "IGKV10S12": 170,  # 0.008368101976644135 of entries %
        "IGKV14S14": 169,  # 0.008318877847369758 of entries %
        "IGKV4-61": 165,  # 0.008121981330272249 of entries %
        "IGKV5S5": 164,  # 0.008072757200997871 of entries %
        "IGLV3-16": 162,  # 0.007974308942449116 of entries %
        "IGKV8S7": 156,  # 0.007678964166802854 of entries %
        "IGKV14S4": 151,  # 0.007432843520430967 of entries %
        "IGKV4-74": 145,  # 0.0071374987447847035 of entries %
        "IGKV22S9": 142,  # 0.006989826356961572 of entries %
        "IGKV1-88": 140,  # 0.006891378098412817 of entries %
        "IGKV4-80": 139,  # 0.00684215396913844 of entries %
        "IGKV5S6": 138,  # 0.0067929298398640625 of entries %
        "IGKV4-70": 137,  # 0.006743705710589685 of entries %
        "IGKV9S1": 133,  # 0.006546809193492176 of entries %
        "IGKV14-100": 107,  # 0.005266981832358368 of entries %
        "IGKV6-14": 106,  # 0.00521775770308399 of entries %
        "IGKV12-89": 105,  # 0.005168533573809613 of entries %
        "IGKV19S1": 105,  # 0.005168533573809613 of entries %
        "IGLV4-3": 101,  # 0.004971637056712104 of entries %
    },
    "l_j_call": {  # Total number of categories 13
        "IGKJ1": 394963,  # 19.441709770595867 of entries %
        "IGLJ2": 331887,  # 16.336848592485246 of entries %
        "IGKJ2": 318426,  # 15.674242588322855 of entries %
        "IGKJ4": 286135,  # 14.084746229923939 of entries %
        "IGLJ3": 279366,  # 13.75154809886568 of entries %
        "IGLJ1": 148743,  # 7.321744660658697 of entries %
        "IGKJ5": 124786,  # 6.142482195632441 of entries %
        "IGKJ3": 123017,  # 6.0554047109460685 of entries %
        "IGKJ2-3": 11629,  # 0.5724273993317333 of entries %
        "IGKJ2-1": 6513,  # 0.32059675396401915 of entries %
        "IGLJ7": 5155,  # 0.25375038640941483 of entries %
        "IGKJ2-2": 871,  # 0.0428742165979826 of entries %
    },
}

VALID_ALLELE_COUNTS = {
    "h_v_call": {  # Total number of categories 544
        "IGHV3-23*01": 178706,  # 8.796647246106863 of entries %
        "IGHV4-39*01": 102458,  # 5.043405837194146 of entries %
        "IGHV4-59*01": 72591,  # 3.57322876815632 of entries %
        "IGHV4-34*01": 68670,  # 3.380220957271487 of entries %
        "IGHV3-21*01": 63212,  # 3.111555659691936 of entries %
        "IGHV5-51*01": 58781,  # 2.89344354287717 of entries %
        "IGHV3-15*01": 51844,  # 2.551975758100815 of entries %
        "IGHV3-30*18": 51595,  # 2.539718949911495 of entries %
        "IGHV3-7*01": 49743,  # 2.4485558624953483 of entries %
        "IGHV1-18*01": 47959,  # 2.3607400158698595 of entries %
        "IGHV3-9*01": 45766,  # 2.25279150037115 of entries %
        "IGHV1-69*01": 43381,  # 2.1353919520517604 of entries %
        "IGHV3-33*01": 42172,  # 2.075879979759038 of entries %
        "IGHV3-74*01": 42050,  # 2.069874635987564 of entries %
        "IGHV1-46*01": 38261,  # 1.8833644101669487 of entries %
        "IGHV4-61*02": 37328,  # 1.8374382975539545 of entries %
        "IGHV1-2*02": 37307,  # 1.8364045908391926 of entries %
        "IGHV3-23*04": 35516,  # 1.748244175308783 of entries %
        "IGHV3-48*03": 34437,  # 1.6951313398217298 of entries %
        "IGHV4-31*03": 32126,  # 1.5813743770686441 of entries %
        "IGHV3-11*01": 31243,  # 1.5379094709193688 of entries %
        "IGHV4-4*02": 30679,  # 1.5101470620086201 of entries %
        "IGHV1-3*01": 28222,  # 1.3892033763814753 of entries %
        "IGHV3-30-3*01": 27874,  # 1.3720733793939919 of entries %
        "IGHV3-7*03": 26007,  # 1.2801719300387295 of entries %
        "IGHV2-5*02": 25781,  # 1.2690472768227203 of entries %
        "IGHV3-30*02": 23673,  # 1.165282812312333 of entries %
        "IGHV3-48*02": 22148,  # 1.0902160151689078 of entries %
        "IGHV1-8*01": 22008,  # 1.0833246370704948 of entries %
        "IGHV4-59*08": 20567,  # 1.0123926667861172 of entries %
        "IGHV3-30*04": 20315,  # 0.9999881862089741 of entries %
        "IGHV6-1*01": 20033,  # 0.9861069817535998 of entries %
        "IGHV7-4-1*02": 19593,  # 0.9644483648728738 of entries %
        "IGHV3-48*01": 18647,  # 0.9178823385793129 of entries %
        "IGHV4-38-2*01": 17786,  # 0.875500363274074 of entries %
        "IGHV3-53*01": 17016,  # 0.8375977837328036 of entries %
        "IGHV1-2*06": 16345,  # 0.8045683929896964 of entries %
        "IGHV2-5*01": 15987,  # 0.7869461547094694 of entries %
        "IGHV5-10-1*03": 15468,  # 0.7613988316160676 of entries %
        "IGHV1-24*01": 14497,  # 0.7136022020906472 of entries %
        "IGHV4-4*07": 14154,  # 0.6967183257495358 of entries %
        "IGHV5-51*03": 14153,  # 0.6966691016202614 of entries %
        "IGHV3-49*04": 14011,  # 0.6896792752632999 of entries %
        "IGHV1-18*04": 13689,  # 0.6738291056369504 of entries %
        "IGHV2-26*01": 12887,  # 0.6343513539588999 of entries %
        "IGHV4-38-2*02": 11341,  # 0.5582508501007125 of entries %
        "IGHV4-61*01": 10982,  # 0.5405793876912112 of entries %
        "IGHV3-72*01": 10961,  # 0.5395456809764492 of entries %
        "IGHV3-49*03": 10011,  # 0.4927827581657908 of entries %
        "IGHV4-39*07": 9508,  # 0.46802302114077904 of entries %
        "IGHV3-33*08": 9346,  # 0.46004871219832993 of entries %
        "IGHV3-30*03": 9216,  # 0.4536495753926609 of entries %
        "IGHV3-11*06": 9119,  # 0.4488748348530463 of entries %
        "IGHV3-66*02": 8881,  # 0.4371594920857445 of entries %
        "IGHV4-30-4*01": 8371,  # 0.4120551861558121 of entries %
        "IGHV1-69*05": 7601,  # 0.3741526066145416 of entries %
        "IGHV3-66*01": 7487,  # 0.3685410558772626 of entries %
        "IGHV2-70*01": 7128,  # 0.35086959346776114 of entries %
        "IGHV3-73*02": 6759,  # 0.33270588976551596 of entries %
        "IGHV1-69*09": 6665,  # 0.3280788216137245 of entries %
        "IGHV3-13*01": 6560,  # 0.32291028803991484 of entries %
        "IGHV1-69*02": 6368,  # 0.3134592552192344 of entries %
        "IGHV3-53*02": 6322,  # 0.3111949452726131 of entries %
        "IGHV3-7*05": 6168,  # 0.30361442936435895 of entries %
        "IGHV4-30-2*01": 5883,  # 0.28958555252116147 of entries %
        "IGHV3-43*01": 5861,  # 0.28850262167712515 of entries %
        "IGHV3-49*05": 5475,  # 0.2695021077772155 of entries %
        "IGHV2-70*15": 5392,  # 0.26541650504744224 of entries %
        "IGHV1-2*04": 5328,  # 0.2622661607738821 of entries %
        "IGHV3-64*01": 4859,  # 0.23918004414419913 of entries %
        "IGHV3-48*04": 4762,  # 0.23440530360458453 of entries %
        "IGHV3-21*02": 4624,  # 0.2276123737647205 of entries %
        "IGHV4-39*02": 4519,  # 0.22244384019091087 of entries %
        "IGHV3-64D*06": 4508,  # 0.2219023747688927 of entries %
        "IGHV3-73*01": 4487,  # 0.2208686680541308 of entries %
        "IGHV1-46*03": 4220,  # 0.20772582553787206 of entries %
        "IGHV5-25*01": 4118,  # 0.2027049643518856 of entries %
        "IGHV4-34*02": 3954,  # 0.19463220715088772 of entries %
        "IGHV1-69*08": 3817,  # 0.18788850144029803 of entries %
        "IGHV5-34*01": 3608,  # 0.17760065842195316 of entries %
        "IGHV1-46*04": 3523,  # 0.1734166074336311 of entries %
        "IGHV3-43D*04": 3259,  # 0.1604214373051955 of entries %
        "IGHV3-43*02": 3177,  # 0.15638505870469657 of entries %
        "IGHV1-58*01": 3084,  # 0.15180721468217948 of entries %
        "IGHV3-20*01": 3084,  # 0.15180721468217948 of entries %
        "IGHV1-69-2*01": 3073,  # 0.15126574926016134 of entries %
        "IGHV1-3*04": 3065,  # 0.15087195622596633 of entries %
        "IGHV1-69*04": 2982,  # 0.146786353496193 of entries %
        "IGHV3-74*03": 2904,  # 0.14294687141279158 of entries %
        "IGHV3-53*04": 2900,  # 0.14274997489569408 of entries %
        "IGHV1-69*06": 2842,  # 0.1398949753977802 of entries %
        "IGHV3-64D*09": 2610,  # 0.12847497740612465 of entries %
        "IGHV3-7*04": 2500,  # 0.12306032318594316 of entries %
        "IGHV3-21*06": 2418,  # 0.11902394458544423 of entries %
        "IGHV4-59*11": 2403,  # 0.11828558264632857 of entries %
        "IGHV3-1*01": 2249,  # 0.11070506673807447 of entries %
        "IGHV9-4*01": 2121,  # 0.10440437819095418 of entries %
        "IGHV5-7*01": 2073,  # 0.10204161998578407 of entries %
        "IGHV4-59*02": 2065,  # 0.10164782695158905 of entries %
        "IGHV8-20*01": 1951,  # 0.09603627621431005 of entries %
        "IGHV1-58*02": 1906,  # 0.09382119039696307 of entries %
        "IGHV2-63*01": 1640,  # 0.08072757200997871 of entries %
        "IGHV5-29*01": 1633,  # 0.08038300310505807 of entries %
        "IGHV2-1*01": 1398,  # 0.06881533272557942 of entries %
        "IGHV2-70*20": 1368,  # 0.0673386088473481 of entries %
        "IGHV4-59*12": 1342,  # 0.06605878148621429 of entries %
        "IGHV3-30*01": 1281,  # 0.06305610960047728 of entries %
        "IGHV5-31*01": 1280,  # 0.0630068854712029 of entries %
        "IGHV3-15*05": 1252,  # 0.061628609851520334 of entries %
        "IGHV1-28*01": 1250,  # 0.06153016159297158 of entries %
        "IGHV1-82*01": 1244,  # 0.061234816817325316 of entries %
        "IGHV3-33*06": 1215,  # 0.05980731706836838 of entries %
        "IGHV3-64D*08": 1215,  # 0.05980731706836838 of entries %
        "IGHV3-4*01": 1210,  # 0.05956119642199649 of entries %
        "IGHV3-15*07": 1209,  # 0.059511972292722114 of entries %
        "IGHV11-4*01": 1196,  # 0.058872058612155206 of entries %
        "IGHV3-43D*03": 1191,  # 0.058625937965783324 of entries %
        "IGHV1-43*01": 1175,  # 0.05783835189739329 of entries %
        "IGHV3-23*03": 1172,  # 0.05769067950957016 of entries %
        "IGHV2-70*04": 1165,  # 0.05734611060464952 of entries %
        "IGHV4-2*01": 1131,  # 0.055672490209320684 of entries %
        "IGHV3-20*04": 1116,  # 0.05493412827020503 of entries %
        "IGHV1-38*01": 1030,  # 0.05070085315260858 of entries %
        "IGHV4-61*08": 994,  # 0.048928784498731004 of entries %
        "IGHV5-54*01": 970,  # 0.04774740539614595 of entries %
        "IGHV3-2*02": 932,  # 0.04587688848371961 of entries %
        "IGHV5-62*01": 904,  # 0.04449861286403705 of entries %
        "IGHV1-26*01": 893,  # 0.043957147442018896 of entries %
        "IGHV3-11*05": 892,  # 0.04390792331274452 of entries %
        "IGHV1-9*01": 891,  # 0.04385869918347014 of entries %
        "IGHV1-46*02": 861,  # 0.04238197530523882 of entries %
        "IGHV7-7*01": 858,  # 0.04223430291741569 of entries %
        "IGHV1-36*01": 811,  # 0.03992076884151996 of entries %
        "IGHV4-59*13": 809,  # 0.03982232058297121 of entries %
        "IGHV2-65*01": 800,  # 0.03937930341950181 of entries %
        "IGHV1-61*01": 783,  # 0.0385424932218374 of entries %
        "IGHV3-13*05": 771,  # 0.03795180367054487 of entries %
        "IGHV5S23*01": 761,  # 0.0374595623778011 of entries %
        "IGHV4-4*09": 726,  # 0.035736717853197895 of entries %
        "IGHV3-30*09": 715,  # 0.03519525243117975 of entries %
        "IGHV3-23*05": 701,  # 0.034506114621338464 of entries %
        "IGHV3-15*02": 697,  # 0.03430921810424095 of entries %
        "IGHV3-33*03": 691,  # 0.03401387332859469 of entries %
        "IGHV5-20*01": 686,  # 0.033767752682222804 of entries %
        "IGHV2-70*13": 685,  # 0.03371852855294843 of entries %
        "IGHV10-5*01": 681,  # 0.033521632035850915 of entries %
        "IGHV1-6*01": 671,  # 0.033029390743107144 of entries %
        "IGHV3-33*05": 662,  # 0.03258637357963775 of entries %
        "IGHV4-31*11": 645,  # 0.031749563381973336 of entries %
        "IGHV2-64*01": 638,  # 0.031404994477052695 of entries %
        "IGHV2-9-1*01": 636,  # 0.03130654621850394 of entries %
        "IGHV9-3*01": 630,  # 0.031011201442857676 of entries %
        "IGHV1-42*01": 627,  # 0.030863529055034547 of entries %
        "IGHV5-19*01": 612,  # 0.030125167115918887 of entries %
        "IGHV2-47*01": 603,  # 0.029682149952449492 of entries %
        "IGHV1-55*01": 576,  # 0.028353098462041305 of entries %
        "IGHV8S18*01": 575,  # 0.02830387433276693 of entries %
        "IGHV3-64*07": 567,  # 0.02791008129857191 of entries %
        "IGHV3-30*14": 554,  # 0.027270167618005006 of entries %
        "IGHV1-11*01": 553,  # 0.027220943488730626 of entries %
        "IGHV7-4-1*01": 550,  # 0.027073271100907497 of entries %
        "IGHV14-4*01": 548,  # 0.02697482284235874 of entries %
        "IGHV1-39*01": 534,  # 0.02628568503251746 of entries %
        "IGHV5-10-1*01": 532,  # 0.026187236773968704 of entries %
        "IGHV1-69*18": 529,  # 0.026039564386145575 of entries %
        "IGHV1-8*02": 511,  # 0.025153530059206782 of entries %
        "IGHV4-61*03": 505,  # 0.02485818528356052 of entries %
        "IGHV3-15*04": 503,  # 0.024759737025011764 of entries %
        "IGHV6-8*01": 493,  # 0.024267495732267993 of entries %
        "IGHV7-3*02": 490,  # 0.02411982334444486 of entries %
        "IGHV2-9*02": 484,  # 0.023824478568798598 of entries %
        "IGHV2-70*17": 480,  # 0.02362758205170109 of entries %
        "IGHV3-6*01": 467,  # 0.02298766837113418 of entries %
        "IGHV1S81*02": 466,  # 0.022938444241859805 of entries %
        "IGHV2S18*01": 462,  # 0.022741547724762296 of entries %
        "IGHV3-6*02": 454,  # 0.022347754690567278 of entries %
        "IGHV5S13*01": 452,  # 0.022249306432018525 of entries %
        "IGHV1-69*12": 451,  # 0.022200082302744145 of entries %
        "IGHV1-13*01": 449,  # 0.022101634044195392 of entries %
        "IGHV1-14*01": 444,  # 0.021855513397823507 of entries %
        "IGHV1-25*01": 442,  # 0.02175706513927475 of entries %
        "IGHV8-17*01": 440,  # 0.021658616880725998 of entries %
        "IGHV3-9*02": 423,  # 0.020821806683061585 of entries %
        "IGHV5-17*02": 416,  # 0.020477237778140943 of entries %
        "IGHV1-47*01": 413,  # 0.02032956539031781 of entries %
        "IGHV6-6*01": 412,  # 0.020280341261043434 of entries %
        "IGHV1-31*01": 409,  # 0.0201326688732203 of entries %
        "IGHV1-2*05": 407,  # 0.020034220614671548 of entries %
        "IGHV1-45*02": 402,  # 0.01978809996829966 of entries %
        "IGHV5-46*01": 401,  # 0.019738875839025283 of entries %
        "IGHV3-21*04": 399,  # 0.01964042758047653 of entries %
        "IGHV8-8*01": 398,  # 0.01959120345120215 of entries %
        "IGHV1-65*01": 396,  # 0.019492755192653397 of entries %
        "IGHV3-11*04": 394,  # 0.01939430693410464 of entries %
        "IGHV6-22*01": 390,  # 0.019197410417007135 of entries %
        "IGHV1-72*01": 369,  # 0.01816370370224521 of entries %
        "IGHV4-30-4*08": 368,  # 0.018114479572970833 of entries %
        "IGHV1-7*01": 367,  # 0.018065255443696457 of entries %
        "IGHV8-28*01": 367,  # 0.018065255443696457 of entries %
        "IGHV5-58*01": 366,  # 0.01801603131442208 of entries %
        "IGHV3-64*02": 363,  # 0.017868358926598948 of entries %
        "IGHV5-4*02": 363,  # 0.017868358926598948 of entries %
        "IGHV3-30*19": 358,  # 0.017622238280227062 of entries %
        "IGHV5-17*01": 356,  # 0.017523790021678306 of entries %
        "IGHV4-4*08": 352,  # 0.017326893504580797 of entries %
        "IGHV3-66*04": 347,  # 0.01708077285820891 of entries %
        "IGHV2-2*02": 347,  # 0.01708077285820891 of entries %
        "IGHV9-2-1*01": 344,  # 0.01693310047038578 of entries %
        "IGHV6-6*02": 343,  # 0.016883876341111402 of entries %
        "IGHV4-31*02": 342,  # 0.016834652211837026 of entries %
        "IGHV5-22*01": 341,  # 0.01678542808256265 of entries %
        "IGHV8-23*01": 339,  # 0.016686979824013893 of entries %
        "IGHV2-8*01": 331,  # 0.016293186789818875 of entries %
        "IGHV14-3*02": 328,  # 0.016145514401995742 of entries %
        "IGHV9-3-1*01": 313,  # 0.015407152462880084 of entries %
        "IGHV5S14*01": 310,  # 0.015259480075056953 of entries %
        "IGHV4-59*03": 309,  # 0.015210255945782574 of entries %
        "IGHV1-57*01": 309,  # 0.015210255945782574 of entries %
        "IGHV2S30*01": 300,  # 0.01476723878231318 of entries %
        "IGHV5-35*01": 298,  # 0.014668790523764425 of entries %
        "IGHV11-3*01": 296,  # 0.01457034226521567 of entries %
        "IGHV5-50*01": 293,  # 0.01442266987739254 of entries %
        "IGHV2-5*05": 292,  # 0.014373445748118161 of entries %
        "IGHV4-28*01": 292,  # 0.014373445748118161 of entries %
        "IGHV1-69*10": 291,  # 0.014324221618843785 of entries %
        "IGHV8-12*01": 291,  # 0.014324221618843785 of entries %
        "IGHV2-41*01": 289,  # 0.01422577336029503 of entries %
        "IGHV2-32*01": 284,  # 0.013979652713923143 of entries %
        "IGHV5-6*01": 279,  # 0.013733532067551258 of entries %
        "IGHV3-1*02": 278,  # 0.01368430793827688 of entries %
        "IGHV1-69*17": 273,  # 0.013438187291904994 of entries %
        "IGHV3-30-3*02": 268,  # 0.013192066645533107 of entries %
        "IGHV1-74*01": 267,  # 0.01314284251625873 of entries %
        "IGHV2S13*01": 266,  # 0.013093618386984352 of entries %
        "IGHV2-16*01": 264,  # 0.012995170128435598 of entries %
        "IGHV1S56*01": 259,  # 0.012749049482063712 of entries %
        "IGHV6-13*01": 259,  # 0.012749049482063712 of entries %
        "IGHV4-31*01": 253,  # 0.012453704706417448 of entries %
        "IGHV3-8*02": 249,  # 0.01225680818931994 of entries %
        "IGHV3-13*04": 247,  # 0.012158359930771185 of entries %
        "IGHV1-64*01": 247,  # 0.012158359930771185 of entries %
        "IGHV5-9-1*01": 246,  # 0.012109135801496806 of entries %
        "IGHV1-15*01": 246,  # 0.012109135801496806 of entries %
        "IGHV9-1*01": 240,  # 0.011813791025850544 of entries %
        "IGHV2-3*01": 238,  # 0.01171534276730179 of entries %
        "IGHV1-53*01": 236,  # 0.011616894508753035 of entries %
        "IGHV1-59*01": 234,  # 0.01151844625020428 of entries %
        "IGHV2-72*01": 231,  # 0.011370773862381148 of entries %
        "IGHV5-6-5*01": 228,  # 0.011223101474558017 of entries %
        "IGHV5S10*01": 228,  # 0.011223101474558017 of entries %
        "IGHV9-6*01": 228,  # 0.011223101474558017 of entries %
        "IGHV2-30*01": 225,  # 0.011075429086734884 of entries %
        "IGHV1-16*01": 223,  # 0.01097698082818613 of entries %
        "IGHV1-5*01": 219,  # 0.01078008431108862 of entries %
        "IGHV4-34*12": 215,  # 0.010583187793991112 of entries %
        "IGHV1-50*01": 215,  # 0.010583187793991112 of entries %
        "IGHV4-1*02": 213,  # 0.010484739535442357 of entries %
        "IGHV2-2*01": 212,  # 0.01043551540616798 of entries %
        "IGHV2-43*01": 210,  # 0.010337067147619226 of entries %
        "IGHV7-6*01": 208,  # 0.010238618889070471 of entries %
        "IGHV2-5*04": 204,  # 0.010041722371972962 of entries %
        "IGHV5-27*01": 197,  # 0.00969715346705232 of entries %
        "IGHV1-60*01": 194,  # 0.00954948107922919 of entries %
        "IGHV2-6-7*01": 193,  # 0.009500256949954813 of entries %
        "IGHV1S135*01": 192,  # 0.009451032820680435 of entries %
        "IGHV6-14*01": 190,  # 0.00935258456213168 of entries %
        "IGHV4-30-2*06": 182,  # 0.008958791527936662 of entries %
        "IGHV1-87*01": 180,  # 0.008860343269387908 of entries %
        "IGHV1-19*01": 177,  # 0.008712670881564777 of entries %
        "IGHV1S29*02": 176,  # 0.008663446752290398 of entries %
        "IGHV8-22*01": 173,  # 0.008515774364467267 of entries %
        "IGHV1-35*01": 172,  # 0.00846655023519289 of entries %
        "IGHV1-12*01": 171,  # 0.008417326105918513 of entries %
        "IGHV2-52*01": 170,  # 0.008368101976644135 of entries %
        "IGHV6-1*02": 168,  # 0.00826965371809538 of entries %
        "IGHV2-77*01": 167,  # 0.008220429588821004 of entries %
        "IGHV1S137*01": 166,  # 0.008171205459546626 of entries %
        "IGHV9-3*02": 165,  # 0.008121981330272249 of entries %
        "IGHV2S63*01": 165,  # 0.008121981330272249 of entries %
        "IGHV1-75*01": 164,  # 0.008072757200997871 of entries %
        "IGHV3-7*02": 163,  # 0.008023533071723495 of entries %
        "IGHV1-54*01": 161,  # 0.00792508481317474 of entries %
        "IGHV1-80*01": 161,  # 0.00792508481317474 of entries %
        "IGHV6-3*01": 160,  # 0.007875860683900362 of entries %
        "IGHV1-76*01": 159,  # 0.007826636554625985 of entries %
        "IGHV1-69*13": 157,  # 0.007728188296077231 of entries %
        "IGHV1S22*01": 155,  # 0.007629740037528476 of entries %
        "IGHV10-1*02": 154,  # 0.007580515908254099 of entries %
        "IGHV14-4*02": 152,  # 0.0074820676497053444 of entries %
        "IGHV2-61*01": 151,  # 0.007432843520430967 of entries %
        "IGHV11-2*01": 149,  # 0.007334395261882213 of entries %
        "IGHV1-32*01": 147,  # 0.007235947003333458 of entries %
        "IGHV9-1*02": 146,  # 0.007186722874059081 of entries %
        "IGHV2-26*02": 145,  # 0.0071374987447847035 of entries %
        "IGHV5-12-1*01": 145,  # 0.0071374987447847035 of entries %
        "IGHV8-31*01": 145,  # 0.0071374987447847035 of entries %
        "IGHV1-52*01": 143,  # 0.007039050486235949 of entries %
        "IGHV4-59*07": 141,  # 0.006940602227687194 of entries %
        "IGHV3-74*02": 140,  # 0.006891378098412817 of entries %
        "IGHV1-29*01": 137,  # 0.006743705710589685 of entries %
        "IGHV5-9-4*01": 134,  # 0.006596033322766553 of entries %
        "IGHV1-81*01": 132,  # 0.006497585064217799 of entries %
        "IGHV8-16*01": 128,  # 0.00630068854712029 of entries %
        "IGHV1-69*11": 127,  # 0.006251464417845912 of entries %
        "IGHV3-3*01": 125,  # 0.006153016159297158 of entries %
        "IGHV4-30-4*07": 122,  # 0.006005343771474027 of entries %
        "IGHV6-5*01": 122,  # 0.006005343771474027 of entries %
        "IGHV14-2*01": 122,  # 0.006005343771474027 of entries %
        "IGHV1-2*07": 120,  # 0.005906895512925272 of entries %
        "IGHV2-45*01": 118,  # 0.005808447254376518 of entries %
        "IGHV3-21*03": 116,  # 0.005709998995827763 of entries %
        "IGHV5-9-3*01": 115,  # 0.005660774866553386 of entries %
        "IGHV10S3*01": 115,  # 0.005660774866553386 of entries %
        "IGHV7-1*02": 113,  # 0.005562326608004631 of entries %
        "IGHV3-64*05": 112,  # 0.005513102478730254 of entries %
        "IGHV5-6-4*01": 109,  # 0.005365430090907122 of entries %
        "IGHV4-28*07": 104,  # 0.005119309444535236 of entries %
        "IGHV10-10*01": 100,  # 0.004922412927437727 of entries %
    },
    "h_d_call": {  # Total number of categories 72
        "IGHD3-10*01": 207158,  # 10.197172172221446 of entries %
        "IGHD3-22*01": 173978,  # 8.563915562897607 of entries %
        "IGHD6-19*01": 152860,  # 7.524400400881309 of entries %
        "IGHD6-13*01": 135563,  # 6.672970636822405 of entries %
        "IGHD2-15*01": 128770,  # 6.33859112666156 of entries %
        "IGHD1-26*01": 125899,  # 6.197268651514824 of entries %
        "IGHD4-17*01": 117962,  # 5.806576737464091 of entries %
        "IGHD3-3*01": 117363,  # 5.777091484028739 of entries %
        "IGHD2-2*01": 99978,  # 4.92132999659369 of entries %
        "IGHD5-12*01": 85313,  # 4.199458140784948 of entries %
        "IGHD3-16*01": 69192,  # 3.4059159527527116 of entries %
        "IGHD5-18*01": 68381,  # 3.365995183911192 of entries %
        "IGHD1-1*01": 61128,  # 3.0089725742841336 of entries %
        "IGHD3-9*01": 59086,  # 2.908456902305855 of entries %
        "IGHD6-6*01": 47582,  # 2.342182519133419 of entries %
        "IGHD2-21*02": 40214,  # 1.9794991346398074 of entries %
        "IGHD2-21*01": 27625,  # 1.359816571204672 of entries %
        "IGHD2-8*01": 26812,  # 1.3197973541046033 of entries %
        "IGHD3-16*02": 26037,  # 1.281648653916961 of entries %
        "IGHD4-4*01": 25507,  # 1.255559865401541 of entries %
        "IGHD1-7*01": 21734,  # 1.0698372256493154 of entries %
        "IGHD7-27*01": 18921,  # 0.9313697500004923 of entries %
        "IGHD2-2*02": 13430,  # 0.6610800561548866 of entries %
        "IGHD1-20*01": 13394,  # 0.659307987501009 of entries %
        "IGHD2-8*02": 11595,  # 0.5707537789364044 of entries %
        "IGHD6-25*01": 10608,  # 0.522169563342594 of entries %
        "IGHD1-11*01": 6537,  # 0.3217781330666042 of entries %
        "IGHD1-2*01": 5985,  # 0.2946064137071479 of entries %
        "IGHD3-10*02": 5650,  # 0.27811633040023154 of entries %
        "IGHD3-3*02": 4844,  # 0.23844168220508347 of entries %
        "IGHD5-1*01": 4233,  # 0.20836573921843896 of entries %
        "IGHD1-6*01": 3886,  # 0.19128496636023004 of entries %
        "IGHD1-4*01": 3572,  # 0.17582858976807558 of entries %
        "IGHD1-12*02": 3433,  # 0.16898643579893716 of entries %
        "IGHD2-2*03": 3231,  # 0.15904316168551294 of entries %
        "IGHD2-4*01": 3229,  # 0.1589447134269642 of entries %
        "IGHD1-9*01": 3216,  # 0.15830479974639727 of entries %
        "IGHD4-3*01": 2654,  # 0.13064083909419727 of entries %
        "IGHD1-3*01": 2510,  # 0.12355256447868694 of entries %
        "IGHD1-10*01": 2506,  # 0.12335566796158942 of entries %
        "IGHD2-1*01": 2428,  # 0.119516185878188 of entries %
        "IGHD4-1*01": 2274,  # 0.1119356699699339 of entries %
        "IGHD2-3*01": 1843,  # 0.0907200702526773 of entries %
        "IGHD1-12*03": 1786,  # 0.08791429488403779 of entries %
        "IGHD2-5*01": 1416,  # 0.06970136705251821 of entries %
        "IGHD2-14*01": 1326,  # 0.06527119541782425 of entries %
        "IGHD1-5*01": 1192,  # 0.0586751620950577 of entries %
        "IGHD3-1*01": 1175,  # 0.05783835189739329 of entries %
        "IGHD1-12*01": 717,  # 0.0352937006897285 of entries %
        "IGHD2-10*02": 701,  # 0.034506114621338464 of entries %
        "IGHD2-12*01": 608,  # 0.029928270598821378 of entries %
        "IGHD1-1*02": 589,  # 0.02899301214260821 of entries %
        "IGHD4-2*01": 480,  # 0.02362758205170109 of entries %
        "IGHD4-6*01": 401,  # 0.019738875839025283 of entries %
        "IGHD3-2*01": 393,  # 0.019345082804830264 of entries %
        "IGHD2-10*01": 314,  # 0.015456376592154462 of entries %
        "IGHD3-8*01": 309,  # 0.015210255945782574 of entries %
        "IGHD3-4*01": 244,  # 0.012010687542948054 of entries %
        "IGHD3-2*02": 232,  # 0.011419997991655526 of entries %
        "IGHD4-1*02": 191,  # 0.009401808691406058 of entries %
        "IGHD1-8*01": 183,  # 0.00900801565721104 of entries %
    },
    "h_j_call": {  # Total number of categories 21
        "IGHJ4*02": 959102,  # 47.210960835313784 of entries %
        "IGHJ6*02": 259400,  # 12.768739133773463 of entries %
        "IGHJ5*02": 226547,  # 11.151578814722347 of entries %
        "IGHJ3*02": 213703,  # 10.519344098322245 of entries %
        "IGHJ6*03": 151381,  # 7.4515979136845045 of entries %
        "IGHJ2*01": 91445,  # 4.501300501495429 of entries %
        "IGHJ3*01": 50659,  # 2.493645164910678 of entries %
        "IGHJ1*01": 34881,  # 1.7169868532195534 of entries %
        "IGHJ4*01": 22493,  # 1.1071983397685679 of entries %
        "IGHJ5*01": 10827,  # 0.5329496476536827 of entries %
        "IGHJ6*04": 7830,  # 0.385424932218374 of entries %
        "IGHJ1*03": 1641,  # 0.0807767961392531 of entries %
        "IGHJ6*01": 1102,  # 0.05424499046036375 of entries %
        "IGHJ4*03": 487,  # 0.023972150956621727 of entries %
    },
    "l_v_call": {  # Total number of categories 357
        "IGKV3-20*01": 193920,  # 9.545543148887239 of entries %
        "IGKV1-39*01": 167873,  # 8.263402253677535 of entries %
        "IGKV4-1*01": 121703,  # 5.990724205079537 of entries %
        "IGKV3-15*01": 109179,  # 5.374241210047235 of entries %
        "IGKV3-11*01": 101902,  # 5.0160372213175926 of entries %
        "IGKV1-5*03": 98341,  # 4.840750096971535 of entries %
        "IGKV2-28*01": 67522,  # 3.3237116568645018 of entries %
        "IGLV1-40*01": 66557,  # 3.2762103721147278 of entries %
        "IGLV3-1*01": 61982,  # 3.0510099806844515 of entries %
        "IGLV1-44*01": 60024,  # 2.954629135565221 of entries %
        "IGKV1-33*01": 57387,  # 2.8248251066686882 of entries %
        "IGLV2-14*03": 55376,  # 2.7258353826979156 of entries %
        "IGLV2-14*01": 54320,  # 2.673854702184173 of entries %
        "IGLV2-8*01": 40226,  # 1.9800898241910998 of entries %
        "IGLV1-47*01": 39747,  # 1.9565114662686731 of entries %
        "IGLV1-51*01": 38915,  # 1.9155569907123913 of entries %
        "IGLV3-19*01": 37774,  # 1.859392259210327 of entries %
        "IGLV3-25*03": 36487,  # 1.7960408048342034 of entries %
        "IGLV2-11*01": 29970,  # 1.4752471543530867 of entries %
        "IGKV1-9*01": 29172,  # 1.4359662991921336 of entries %
        "IGLV3-21*02": 26943,  # 1.3262457150395466 of entries %
        "IGKV1-12*01": 25062,  # 1.233655127874443 of entries %
        "IGLV1-51*02": 24383,  # 1.200231944097141 of entries %
        "IGLV2-23*02": 23743,  # 1.1687285013615394 of entries %
        "IGLV3-21*04": 22553,  # 1.1101517875250304 of entries %
        "IGKV1-8*01": 21512,  # 1.0589094689504037 of entries %
        "IGKV2-30*01": 21245,  # 1.045766626434145 of entries %
        "IGKV1-27*01": 20769,  # 1.0223359408995414 of entries %
        "IGKV1-17*01": 19120,  # 0.9411653517260933 of entries %
        "IGLV2-23*01": 15233,  # 0.7498311612365889 of entries %
        "IGKV1-16*02": 14569,  # 0.7171463393984023 of entries %
        "IGLV4-69*01": 14055,  # 0.6918451369513725 of entries %
        "IGLV3-10*01": 13805,  # 0.6795391046327781 of entries %
        "IGLV7-46*01": 13414,  # 0.6602924700864966 of entries %
        "IGKV2-24*01": 13134,  # 0.646509713889671 of entries %
        "IGKV2-30*02": 12357,  # 0.6082625654434799 of entries %
        "IGLV8-61*01": 11828,  # 0.5822230010573343 of entries %
        "IGKV1-6*01": 10659,  # 0.5246799939355873 of entries %
        "IGLV7-43*01": 9228,  # 0.4542402649439534 of entries %
        "IGLV3-9*01": 8072,  # 0.3973371715027733 of entries %
        "IGKV2D-29*01": 6928,  # 0.3410247676128857 of entries %
        "IGKV1D-12*01": 6495,  # 0.31971071963708037 of entries %
        "IGKV2-29*02": 6462,  # 0.3180863233710259 of entries %
        "IGKV1-NL1*01": 6263,  # 0.3082907216454248 of entries %
        "IGLV6-57*01": 6115,  # 0.30100555051281697 of entries %
        "IGLV6-57*04": 5873,  # 0.2890933112284177 of entries %
        "IGKV1-5*01": 5135,  # 0.2527659038239273 of entries %
        "IGKV1-17*03": 4692,  # 0.23095961455537814 of entries %
        "IGLV2-18*02": 4561,  # 0.2245112536204347 of entries %
        "IGKV1D-8*01": 4539,  # 0.2234283227763984 of entries %
        "IGLV1-36*01": 4362,  # 0.21471565189483363 of entries %
        "IGLV6-57*02": 4282,  # 0.21077772155288346 of entries %
        "IGKV2-40*01": 4173,  # 0.20541229146197634 of entries %
        "IGKV3D-15*01": 3741,  # 0.18414746761544534 of entries %
        "IGKV1-13*02": 3450,  # 0.16982324599660156 of entries %
        "IGKV1-16*01": 3411,  # 0.16790350495490086 of entries %
        "IGLV10-54*04": 2971,  # 0.14624488807417485 of entries %
        "IGLV3-27*01": 2954,  # 0.14540807787651044 of entries %
        "IGKV6-21*01": 2944,  # 0.14491583658376667 of entries %
        "IGKV3D-20*01": 2899,  # 0.1427007507664197 of entries %
        "IGLV9-49*01": 2873,  # 0.1414209234052859 of entries %
        "IGLV4-60*03": 2826,  # 0.13910738932939015 of entries %
        "IGLV10-54*01": 2727,  # 0.1342342005312268 of entries %
        "IGKV2D-29*02": 2599,  # 0.12793351198410652 of entries %
        "IGKV1D-16*01": 2551,  # 0.1255707537789364 of entries %
        "IGLV5-45*03": 2548,  # 0.12542308139111327 of entries %
        "IGLV3-21*01": 2322,  # 0.11429842817510401 of entries %
        "IGKV16S1*01": 2221,  # 0.1093267911183919 of entries %
        "IGKV3S19*01": 2108,  # 0.10376446451038727 of entries %
        "IGLV2-23*03": 2011,  # 0.09898972397077269 of entries %
        "IGLV3-21*03": 1945,  # 0.09574093143866379 of entries %
        "IGKV10-96*01": 1930,  # 0.09500256949954812 of entries %
        "IGKV6-21*02": 1896,  # 0.0933289491042193 of entries %
        "IGKV12S36*01": 1810,  # 0.08909567398662285 of entries %
        "IGKV12S34*01": 1758,  # 0.08653601926435524 of entries %
        "IGKV1D-13*01": 1636,  # 0.0805306754928812 of entries %
        "IGLV1-47*02": 1552,  # 0.07639584863383352 of entries %
        "IGKV6D-21*02": 1477,  # 0.07270403893825522 of entries %
        "IGKV1-12*02": 1470,  # 0.07235947003333458 of entries %
        "IGKV8-30*01": 1431,  # 0.07043972899163387 of entries %
        "IGKV22S2*01": 1428,  # 0.07029205660381073 of entries %
        "IGLV6-57*03": 1349,  # 0.06640335039113493 of entries %
        "IGLV5-45*02": 1325,  # 0.06522197128854988 of entries %
        "IGKV1-117*01": 1307,  # 0.06433593696161109 of entries %
        "IGLV3-25*02": 1273,  # 0.06266231656628225 of entries %
        "IGKV3S10*01": 1259,  # 0.061973178756440976 of entries %
        "IGKV22S4*01": 1238,  # 0.06093947204167906 of entries %
        "IGKV15S2*01": 1215,  # 0.05980731706836838 of entries %
        "IGKV22S1*01": 1179,  # 0.05803524841449079 of entries %
        "IGKV12S16*01": 1164,  # 0.05729688647537514 of entries %
        "IGKV5-2*01": 1038,  # 0.0510946461868036 of entries %
        "IGLV5-37*01": 1033,  # 0.05084852554043171 of entries %
        "IGKV6S10*01": 1031,  # 0.05075007728188296 of entries %
        "IGKV1D-8*02": 1021,  # 0.05025783598913919 of entries %
        "IGKV8S5*01": 996,  # 0.04902723275727976 of entries %
        "IGKV1-110*01": 989,  # 0.048682663852359115 of entries %
        "IGKV2S27*01": 952,  # 0.04686137106920716 of entries %
        "IGKV22S7*01": 939,  # 0.04622145738864025 of entries %
        "IGLV5-45*01": 923,  # 0.045433871320250216 of entries %
        "IGKV3S9*01": 918,  # 0.04518775067387833 of entries %
        "IGKV1S14*01": 909,  # 0.04474473351040893 of entries %
        "IGKV14-111*01": 905,  # 0.04454783699331143 of entries %
        "IGKV1S22*01": 894,  # 0.04400637157129327 of entries %
        "IGLV1*01": 883,  # 0.043464906149275125 of entries %
        "IGLV4-69*02": 875,  # 0.043071113115080106 of entries %
        "IGKV3S11*01": 873,  # 0.042972664856531353 of entries %
        "IGKV8S4*01": 855,  # 0.042086630529592564 of entries %
        "IGKV12-46*01": 836,  # 0.04115137207337939 of entries %
        "IGKV8S6*01": 828,  # 0.04075757903918437 of entries %
        "IGKV12S11*01": 826,  # 0.04065913078063562 of entries %
        "IGKV19S2*01": 818,  # 0.0402653377464406 of entries %
        "IGKV17-121*01": 816,  # 0.04016688948789185 of entries %
        "IGKV12S38*01": 815,  # 0.04011766535861747 of entries %
        "IGKV6S5*01": 809,  # 0.03982232058297121 of entries %
        "IGKV14S9*01": 805,  # 0.0396254240658737 of entries %
        "IGKV6-15*01": 784,  # 0.038591717351111776 of entries %
        "IGKV12S29*01": 758,  # 0.03731188998997797 of entries %
        "IGKV1-135*01": 739,  # 0.036376631533764796 of entries %
        "IGKV3-4*01": 720,  # 0.03544137307755163 of entries %
        "IGKV17-127*01": 695,  # 0.0342107698456922 of entries %
        "IGKV12S31*01": 695,  # 0.0342107698456922 of entries %
        "IGKV22S6*01": 683,  # 0.033620080294399675 of entries %
        "IGKV3S1*01": 675,  # 0.03322628726020466 of entries %
        "IGKV12S32*01": 674,  # 0.03317706313093028 of entries %
        "IGKV10-94*01": 670,  # 0.03298016661383277 of entries %
        "IGKV2S23*01": 644,  # 0.03170033925269896 of entries %
        "IGKV12-44*01": 641,  # 0.031552666864875824 of entries %
        "IGKV6S11*01": 640,  # 0.03150344273560145 of entries %
        "IGKV19-93*01": 630,  # 0.031011201442857676 of entries %
        "IGKV10S5*01": 629,  # 0.0309619773135833 of entries %
        "IGKV5S12*01": 622,  # 0.030617408408662658 of entries %
        "IGKV3S18*01": 621,  # 0.03056818427938828 of entries %
        "IGKV1S21*01": 618,  # 0.03042051189156515 of entries %
        "IGKV12S39*01": 616,  # 0.030322063633016396 of entries %
        "IGKV2S6*01": 610,  # 0.03002671885737013 of entries %
        "IGKV14S1*01": 604,  # 0.02973137408172387 of entries %
        "IGKV4-59*01": 584,  # 0.028746891496236323 of entries %
        "IGKV6-23*01": 583,  # 0.028697667366961947 of entries %
        "IGKV3-10*01": 561,  # 0.027614736522925645 of entries %
        "IGKV1S26*01": 561,  # 0.027614736522925645 of entries %
        "IGKV12S8*01": 543,  # 0.026728702195986855 of entries %
        "IGKV4S7*01": 542,  # 0.02667947806671248 of entries %
        "IGKV12S26*01": 532,  # 0.026187236773968704 of entries %
        "IGKV2S25*01": 531,  # 0.026138012644694328 of entries %
        "IGKV4S5*01": 530,  # 0.02608878851541995 of entries %
        "IGKV8-24*01": 524,  # 0.025793443739773686 of entries %
        "IGKV3-1*01": 517,  # 0.025448874834853048 of entries %
        "IGKV2-137*01": 506,  # 0.024907409412834897 of entries %
        "IGKV16-104*01": 473,  # 0.023283013146780447 of entries %
        "IGLV2-18*01": 471,  # 0.02318456488823169 of entries %
        "IGKV6-17*01": 467,  # 0.02298766837113418 of entries %
        "IGKV14S18*01": 465,  # 0.02288922011258543 of entries %
        "IGKV9-120*01": 460,  # 0.022643099466213543 of entries %
        "IGKV12S30*01": 457,  # 0.02249542707839041 of entries %
        "IGKV10S9*01": 455,  # 0.022396978819841654 of entries %
        "IGKV5-43*01": 453,  # 0.0222985305612929 of entries %
        "IGKV1-6*02": 443,  # 0.021806289268549127 of entries %
        "IGKV2S11*01": 433,  # 0.021314047975805356 of entries %
        "IGKV2S17*01": 431,  # 0.021215599717256603 of entries %
        "IGKV8S9*01": 424,  # 0.02087103081233596 of entries %
        "IGLV5-39*01": 411,  # 0.020231117131769057 of entries %
        "IGKV4-57-1*01": 411,  # 0.020231117131769057 of entries %
        "IGKV1S23*01": 402,  # 0.01978809996829966 of entries %
        "IGLV4-60*02": 400,  # 0.019689651709750906 of entries %
        "IGKV2S26*01": 393,  # 0.019345082804830264 of entries %
        "IGKV17S1*01": 384,  # 0.01890206564136087 of entries %
        "IGKV2D-30*01": 380,  # 0.01870516912426336 of entries %
        "IGKV14S15*01": 379,  # 0.018655944994988984 of entries %
        "IGKV3-2*01": 370,  # 0.01821292783151959 of entries %
        "IGKV13-84*01": 357,  # 0.017573014150952682 of entries %
        "IGKV1D-8*04": 354,  # 0.017425341763129553 of entries %
        "IGKV6S8*01": 353,  # 0.017376117633855173 of entries %
        "IGKV12-41*01": 352,  # 0.017326893504580797 of entries %
        "IGKV4S4*01": 348,  # 0.017129996987483288 of entries %
        "IGKV22S8*01": 344,  # 0.01693310047038578 of entries %
        "IGKV3S6*01": 342,  # 0.016834652211837026 of entries %
        "IGKV5S2*01": 341,  # 0.01678542808256265 of entries %
        "IGKV2S16*01": 340,  # 0.01673620395328827 of entries %
        "IGKV14S13*01": 340,  # 0.01673620395328827 of entries %
        "IGKV3D-7*01": 331,  # 0.016293186789818875 of entries %
        "IGKV3-12*01": 329,  # 0.016194738531270122 of entries %
        "IGKV8-28*01": 324,  # 0.015948617884898233 of entries %
        "IGKV1-17*02": 322,  # 0.01585016962634948 of entries %
        "IGKV4S12*01": 315,  # 0.015505600721428838 of entries %
        "IGKV12S20*01": 314,  # 0.015456376592154462 of entries %
        "IGKV5-48*01": 313,  # 0.015407152462880084 of entries %
        "IGKV13-85*01": 312,  # 0.015357928333605707 of entries %
        "IGKV8-27*01": 310,  # 0.015259480075056953 of entries %
        "IGKV4-57*01": 299,  # 0.014718014653038802 of entries %
        "IGKV4-72*01": 294,  # 0.014471894006666916 of entries %
        "IGKV6-25*01": 286,  # 0.014078100972471898 of entries %
        "IGKV3-7*01": 281,  # 0.013831980326100012 of entries %
        "IGKV3-5*01": 279,  # 0.013733532067551258 of entries %
        "IGKV14S19*01": 278,  # 0.01368430793827688 of entries %
        "IGKV3S8*01": 277,  # 0.013635083809002503 of entries %
        "IGKV4-50*01": 270,  # 0.013290514904081861 of entries %
        "IGKV1S1*01": 269,  # 0.013241290774807485 of entries %
        "IGKV6-32*01": 265,  # 0.013044394257709976 of entries %
        "IGKV3S13*01": 263,  # 0.012945945999161221 of entries %
        "IGKV1S18*01": 259,  # 0.012749049482063712 of entries %
        "IGKV12S17*01": 259,  # 0.012749049482063712 of entries %
        "IGKV2-29*03": 256,  # 0.01260137709424058 of entries %
        "IGKV5S10*01": 253,  # 0.012453704706417448 of entries %
        "IGKV12S24*01": 253,  # 0.012453704706417448 of entries %
        "IGKV4-55*01": 248,  # 0.012207584060045561 of entries %
        "IGKV4S9*01": 238,  # 0.01171534276730179 of entries %
        "IGLV2-14*02": 233,  # 0.011469222120929903 of entries %
        "IGKV12S14*01": 232,  # 0.011419997991655526 of entries %
        "IGKV6S4*01": 231,  # 0.011370773862381148 of entries %
        "IGKV8-21*01": 226,  # 0.011124653216009263 of entries %
        "IGKV4-68*01": 224,  # 0.011026204957460508 of entries %
        "IGLV9-49*03": 221,  # 0.010878532569637375 of entries %
        "IGKV6S9*01": 220,  # 0.010829308440362999 of entries %
        "IGKV1S28*01": 218,  # 0.010730860181814244 of entries %
        "IGKV2S20*01": 218,  # 0.010730860181814244 of entries %
        "IGKV3D-15*03": 216,  # 0.01063241192326549 of entries %
        "IGLV3-22*01": 213,  # 0.010484739535442357 of entries %
        "IGKV5-39*01": 211,  # 0.010386291276893602 of entries %
        "IGKV9-124*01": 205,  # 0.010090946501247339 of entries %
        "IGKV2-109*01": 204,  # 0.010041722371972962 of entries %
        "IGKV6-20*01": 204,  # 0.010041722371972962 of entries %
        "IGKV1S30*01": 198,  # 0.009746377596326699 of entries %
        "IGKV4S18*01": 197,  # 0.00969715346705232 of entries %
        "IGKV4S21*01": 195,  # 0.009598705208503568 of entries %
        "IGKV4-53*01": 194,  # 0.00954948107922919 of entries %
        "IGKV14S8*01": 194,  # 0.00954948107922919 of entries %
        "IGKV3S5*01": 184,  # 0.009057239786485417 of entries %
        "IGKV4-91*01": 180,  # 0.008860343269387908 of entries %
        "IGKV4S2*01": 178,  # 0.008761895010839153 of entries %
        "IGLV5-52*01": 177,  # 0.008712670881564777 of entries %
        "IGKV8-19*01": 177,  # 0.008712670881564777 of entries %
        "IGKV4S13*01": 174,  # 0.008564998493741644 of entries %
        "IGKV10S12*01": 170,  # 0.008368101976644135 of entries %
        "IGKV14S14*01": 169,  # 0.008318877847369758 of entries %
        "IGLV7-46*02": 166,  # 0.008171205459546626 of entries %
        "IGKV4-61*01": 165,  # 0.008121981330272249 of entries %
        "IGKV5S5*01": 164,  # 0.008072757200997871 of entries %
        "IGLV3-16*01": 162,  # 0.007974308942449116 of entries %
        "IGKV8S7*01": 156,  # 0.007678964166802854 of entries %
        "IGKV14S4*01": 151,  # 0.007432843520430967 of entries %
        "IGKV4-74*01": 145,  # 0.0071374987447847035 of entries %
        "IGKV22S9*01": 142,  # 0.006989826356961572 of entries %
        "IGKV1-88*01": 140,  # 0.006891378098412817 of entries %
        "IGKV4-80*01": 139,  # 0.00684215396913844 of entries %
        "IGKV5S6*01": 138,  # 0.0067929298398640625 of entries %
        "IGKV4-70*01": 137,  # 0.006743705710589685 of entries %
        "IGKV3D-11*01": 136,  # 0.006694481581315308 of entries %
        "IGKV9S1*01": 133,  # 0.006546809193492176 of entries %
        "IGKV3-11*02": 128,  # 0.00630068854712029 of entries %
        "IGKV14-100*01": 107,  # 0.005266981832358368 of entries %
        "IGKV6-14*01": 106,  # 0.00521775770308399 of entries %
        "IGKV12-89*01": 105,  # 0.005168533573809613 of entries %
        "IGKV19S1*01": 105,  # 0.005168533573809613 of entries %
        "IGLV4-3*01": 101,  # 0.004971637056712104 of entries %
    },
    "l_j_call": {  # Total number of categories 20
        "IGKJ1*01": 394962,  # 19.441660546466593 of entries %
        "IGLJ2*01": 331887,  # 16.336848592485246 of entries %
        "IGKJ2*01": 299852,  # 14.759953611180572 of entries %
        "IGKJ4*01": 284930,  # 14.025431154148315 of entries %
        "IGLJ3*02": 279307,  # 13.748643875238491 of entries %
        "IGLJ1*01": 148743,  # 7.321744660658697 of entries %
        "IGKJ5*01": 124786,  # 6.142482195632441 of entries %
        "IGKJ3*01": 123017,  # 6.0554047109460685 of entries %
        "IGKJ2-3*01": 11629,  # 0.5724273993317333 of entries %
        "IGKJ2*02": 9433,  # 0.46433121144520073 of entries %
        "IGKJ2-1*01": 6513,  # 0.32059675396401915 of entries %
        "IGKJ2*03": 5884,  # 0.28963477665043585 of entries %
        "IGLJ7*01": 5150,  # 0.2535042657630429 of entries %
        "IGKJ2*04": 3257,  # 0.16032298904664674 of entries %
        "IGKJ4*02": 1205,  # 0.05931507577562461 of entries %
        "IGKJ2-2*01": 871,  # 0.0428742165979826 of entries %
    },
}

VALID_GENE_FAMILIES = {
    "h_v_call": {
        "IGHV3": 971130,
        "IGHV4": 464912,
        "IGHV1": 346456,
        "IGHV5": 111565,
        "IGHV2": 83370,
        "IGHV6": 22805,
        "IGHV7": 22008,
        "IGHV9": 4292,
        "IGHV8": 4981,
    },
    "h_d_call": {
        "IGHD3": 665850,
        "IGHD6": 346722,
        "IGHD2": 363695,
        "IGHD1": 258267,
        "IGHD4": 149499,
        "IGHD5": 157927,
        "IGHD7": 18921,
    },
    "h_j_call": {
        "IGHJ4": 982082,
        "IGHJ6": 419713,
        "IGHJ3": 264362,
        "IGHJ5": 237374,
        "IGHJ2": 91466,
        "IGHJ1": 36522,
    },
    "l_v_call": {
        "IGKV3": 423319,
        "IGKV1": 544888,
        "IGKV4": 128235,
        "IGLV2": 226260,
        "IGKV2": 146583,
        "IGLV1": 242264,
        "IGLV3": 216562,
        "IGLV6": 17619,
        "IGLV4": 18262,
        "IGLV7": 22808,
        "IGLV8": 11840,
        "IGKV6": 12441,
        "IGLV5": 6422,
        "IGLV9": 3147,
        "IGKV8": 6344,
        "IGKV5": 3604,
        "IGKV9": 918,
    },
    "l_j_call": {
        "IGKJ1": 394963,
        "IGLJ2": 331887,
        "IGKJ2": 337439,
        "IGKJ4": 286135,
        "IGLJ3": 279366,
        "IGLJ1": 148743,
        "IGKJ5": 124786,
        "IGKJ3": 123017,
        "IGLJ7": 5155,
    },
}

VALID_REGION_LENGTHS = {
    "h_cdr1_len": {  # Total number of categories 21
        8: 1653241,  # 81.37934870570075 of entries %
        10: 309071,  # 15.213750858961056 of entries %
        9: 64433,  # 3.17165832153595 of entries %
        11: 1270,  # 0.06251464417845913 of entries %
        7: 857,  # 0.04218507878814132 of entries %
        12: 603,  # 0.029682149952449492 of entries %
        6: 600,  # 0.02953447756462636 of entries %
        13: 347,  # 0.01708077285820891 of entries %
        5: 305,  # 0.015013359428685065 of entries %
        15: 238,  # 0.01171534276730179 of entries %
        4: 199,  # 0.009795601725601075 of entries %
        14: 156,  # 0.007678964166802854 of entries %
    },
    "h_cdr2_len": {  # Total number of categories 24
        8: 1273575,  # 62.69074387650626 of entries %
        7: 613944,  # 30.220918327163897 of entries %
        10: 114399,  # 5.63120225250059 of entries %
        9: 23275,  # 1.145693864692447 of entries %
        6: 2468,  # 0.12148539024966527 of entries %
        11: 1182,  # 0.0581830353626841 of entries %
        12: 687,  # 0.033817043396077814 of entries %
        5: 526,  # 0.025891942978656376 of entries %
        13: 513,  # 0.02525202803811924 of entries %
        14: 319,  # 0.015702528156257382 of entries %
        15: 171,  # 0.00841734267937308 of entries %
        16: 110,  # 0.005414664881468063 of entries %
    },
    "h_cdr3_len": {  # Total number of categories 48
        14: 222597,  # 10.957143504088556 of entries %
        15: 215458,  # 10.605732445198777 of entries %
        16: 191418,  # 9.422384377442748 of entries %
        13: 185810,  # 9.14633546047204 of entries %
        17: 165693,  # 8.156093651859392 of entries %
        12: 160773,  # 7.913910935829456 of entries %
        18: 148086,  # 7.289404407725431 of entries %
        11: 124063,  # 6.106893150167067 of entries %
        19: 120281,  # 5.920727493251372 of entries %
        20: 96908,  # 4.770211919721352 of entries %
        10: 75979,  # 3.7400001181379103 of entries %
        21: 71532,  # 3.5211004152547547 of entries %
        22: 52687,  # 2.593471699079115 of entries %
        9: 47757,  # 2.3507967417564353 of entries %
        23: 35648,  # 1.7547417603730007 of entries %
        8: 24852,  # 1.2233180607268237 of entries %
        24: 24312,  # 1.19673703091866 of entries %
        25: 15490,  # 0.7624817624601038 of entries %
        7: 13261,  # 0.6527611783075169 of entries %
        6: 10567,  # 0.5201513740423446 of entries %
        26: 9561,  # 0.470631899992321 of entries %
        5: 5372,  # 0.2644320224619547 of entries %
        27: 5346,  # 0.2631521951008209 of entries %
        28: 3221,  # 0.15855092039276916 of entries %
        29: 1762,  # 0.08673291578145274 of entries %
        30: 1103,  # 0.05429421458963812 of entries %
        31: 651,  # 0.0320449081576196 of entries %
        32: 468,  # 0.02303689250040856 of entries %
        33: 268,  # 0.013192066645533107 of entries %
        4: 153,  # 0.007531291778979722 of entries %
        34: 143,  # 0.007039050486235949 of entries %
        35: 132,  # 0.006497585064217799 of entries %
    },
    "h_fwr1_len": {  # Total number of categories 22
        25: 2001975,  # 98.54547620407142 of entries %
        24: 16602,  # 0.8172189942132113 of entries %
        23: 8999,  # 0.442967939340121 of entries %
        22: 1413,  # 0.06955369466469508 of entries %
        21: 923,  # 0.045433871320250216 of entries %
        26: 884,  # 0.0435141302785495 of entries %
        20: 152,  # 0.0074820676497053444 of entries %
        27: 129,  # 0.006349912676394667 of entries %
        29: 111,  # 0.005463878349455877 of entries %
    },
    "h_fwr2_len": {  # Total number of categories 19
        17: 2027149,  # 99.78464443442459 of entries %
        16: 1707,  # 0.08402558867136199 of entries %
        18: 1522,  # 0.0749191247556022 of entries %
        20: 368,  # 0.018114479572970833 of entries %
        19: 354,  # 0.017425341763129553 of entries %
        21: 124,  # 0.0061037920300227805 of entries %
    },
    "h_fwr3_len": {  # Total number of categories 25
        38: 2026192,  # 99.73753694270901 of entries %
        37: 1821,  # 0.089637139408641 of entries %
        36: 834,  # 0.04105292381483064 of entries %
        40: 672,  # 0.03307861487238152 of entries %
        39: 624,  # 0.030715856667211414 of entries %
        41: 389,  # 0.019148186287732755 of entries %
        42: 230,  # 0.011321549733106772 of entries %
        35: 203,  # 0.009992498242698584 of entries %
        43: 160,  # 0.007875860683900362 of entries %
        44: 146,  # 0.007186722874059081 of entries %
    },
    "h_fwr4_len": {  # Total number of categories 10
        11: 2012699,  # 99.07355083878082 of entries %
        10: 15921,  # 0.7836989052532094 of entries %
        9: 2233,  # 0.10991769709380168 of entries %
        7: 254,  # 0.01250295345357171 of entries %
        8: 165,  # 0.008121997322202095 of entries %
    },
    "l_cdr1_len": {  # Total number of categories 20
        6: 996630,  # 49.05824395872261 of entries %
        9: 333431,  # 16.412850648084888 of entries %
        7: 217814,  # 10.72170449376921 of entries %
        8: 192641,  # 9.482585487545311 of entries %
        11: 144660,  # 7.120762540831415 of entries %
        12: 131285,  # 6.462389811786619 of entries %
        10: 7984,  # 0.3930054481266281 of entries %
        5: 6326,  # 0.3113918417897106 of entries %
        4: 260,  # 0.012798273611338088 of entries %
        13: 174,  # 0.008564998493741644 of entries %
        14: 118,  # 0.005808447254376518 of entries %
    },
    "l_cdr2_len": {  # Total number of categories 15
        3: 2003100,  # 98.60196967660858 of entries %
        7: 24731,  # 1.2173757236644236 of entries %
        8: 3157,  # 0.15540233551447918 of entries %
        2: 319,  # 0.0157026750171425 of entries %
    },
    "l_cdr3_len": {  # Total number of categories 25
        9: 969295,  # 47.71270238500751 of entries %
        10: 421586,  # 20.752203764267612 of entries %
        11: 378346,  # 18.62375241444354 of entries %
        8: 127574,  # 6.279719068049405 of entries %
        12: 94540,  # 4.653649181599627 of entries %
        13: 20595,  # 1.0137709424057997 of entries %
        5: 7065,  # 0.3477684733234754 of entries %
        7: 5539,  # 0.27265245205077565 of entries %
        6: 3711,  # 0.18267074373721404 of entries %
        14: 1819,  # 0.08953869115009225 of entries %
        4: 632,  # 0.031109649701406433 of entries %
        15: 375,  # 0.018459048477891475 of entries %
        16: 171,  # 0.008417326105918513 of entries %
        17: 113,  # 0.005562326608004631 of entries %
    },
    "l_fwr1_len": {  # Total number of categories 19
        26: 1230097,  # 60.55045374802365 of entries %
        25: 790249,  # 38.89931893494736 of entries %
        24: 9346,  # 0.46004871219832993 of entries %
        23: 911,  # 0.04484318176895769 of entries %
        22: 502,  # 0.024710512895737387 of entries %
        21: 149,  # 0.007334395261882213 of entries %
    },
    "l_fwr2_len": {  # Total number of categories 17
        17: 2029542,  # 99.90243777577818 of entries %
        18: 843,  # 0.04149594097830003 of entries %
        16: 672,  # 0.03307861487238152 of entries %
        21: 134,  # 0.006596033322766553 of entries %
        19: 122,  # 0.006005343771474027 of entries %
        20: 103,  # 0.0050700853152608584 of entries %
    },
    "l_fwr3_len": {  # Total number of categories 22
        36: 2005791,  # 98.73331548138245 of entries %
        38: 24537,  # 1.207812460005395 of entries %
        37: 348,  # 0.017129996987483288 of entries %
        35: 332,  # 0.01634241091909325 of entries %
        39: 129,  # 0.006349912676394667 of entries %
    },
    "l_fwr4_len": {  # Total number of categories 10
        10: 1956074,  # 96.2860868422361 of entries %
        9: 67772,  # 3.3360193313095645 of entries %
        8: 3577,  # 0.1760747970857332 of entries %
        7: 3372,  # 0.1659838456173029 of entries %
        5: 530,  # 0.026088801357405256 of entries %
        6: 145,  # 0.007137502258158042 of entries %
    },
}

# Below is a coversion of IMGT indices to the position of that index in the sequence data mode arrays.
# NB: This is rather hacky, with l_fwr1 being a problem.
IMGT2IDX = {
    "h": {
        "h_cdr1": {
            e: i
            for i, e in enumerate(
                [
                    (27, " "),
                    (28, " "),
                    (29, " "),
                    (30, " "),
                    (31, " "),
                    (32, " "),
                    (32, "A"),
                    (32, "B"),
                    (32, "C"),
                    (32, "D"),
                    (32, "E"),
                    (33, "E"),
                    (33, "D"),
                    (33, "C"),
                    (33, "B"),
                    (33, "A"),
                    (33, " "),
                    (34, " "),
                    (35, " "),
                    (36, " "),
                    (37, " "),
                    (38, " "),
                ],
                start=1,
            )
        },
        "h_cdr2": {
            e: i
            for i, e in enumerate(
                [
                    (56, " "),
                    (57, " "),
                    (58, " "),
                    (59, " "),
                    (60, " "),
                    (60, "A"),
                    (60, "B"),
                    (60, "C"),
                    (60, "D"),
                    (60, "E"),
                    (60, "F"),
                    (60, "G"),
                    (61, "H"),
                    (61, "G"),
                    (61, "F"),
                    (61, "E"),
                    (61, "D"),
                    (61, "C"),
                    (61, "B"),
                    (61, "A"),
                    (61, " "),
                    (62, " "),
                    (63, " "),
                    (64, " "),
                    (65, " "),
                ],
                start=1,
            )
        },
        "h_cdr3": {
            e: i
            for i, e in enumerate(
                [
                    (105, " "),
                    (106, " "),
                    (107, " "),
                    (108, " "),
                    (109, " "),
                    (110, " "),
                    (111, " "),
                    (111, "A"),
                    (111, "B"),
                    (111, "C"),
                    (111, "D"),
                    (111, "E"),
                    (111, "F"),
                    (111, "G"),
                    (111, "H"),
                    (111, "I"),
                    (111, "J"),
                    (111, "K"),
                    (111, "L"),
                    (111, "M"),
                    (111, "N"),
                    (111, "O"),
                    (111, "P"),
                    (111, "Q"),
                    (111, "R"),
                    (111, "S"),
                    (111, "T"),
                    (111, "U"),
                    (111, "V"),
                    (112, "W"),
                    (112, "V"),
                    (112, "U"),
                    (112, "T"),
                    (112, "S"),
                    (112, "R"),
                    (112, "Q"),
                    (112, "P"),
                    (112, "O"),
                    (112, "N"),
                    (112, "M"),
                    (112, "L"),
                    (112, "K"),
                    (112, "J"),
                    (112, "I"),
                    (112, "H"),
                    (112, "G"),
                    (112, "F"),
                    (112, "E"),
                    (112, "D"),
                    (112, "C"),
                    (112, "B"),
                    (112, "A"),
                    (112, " "),
                    (113, " "),
                    (114, " "),
                    (115, " "),
                    (116, " "),
                    (117, " "),
                ],
                start=1,
            )
        },
        "h_fwr1": {
            e: i
            for i, e in enumerate(
                [
                    (1, " "),
                    (2, " "),
                    (3, " "),
                    (4, " "),
                    (5, " "),
                    (6, " "),
                    (7, " "),
                    (8, " "),
                    (9, " "),
                    (11, " "),
                    (12, " "),
                    (13, " "),
                    (14, " "),
                    (15, " "),
                    (16, " "),
                    (17, " "),
                    (18, " "),
                    (19, " "),
                    (20, " "),
                    (21, " "),
                    (22, " "),
                    (23, " "),
                    (24, " "),
                    (25, " "),
                    (26, " "),
                ],
                start=1,
            )
        },  # len 25, del @ imgt10
        "h_fwr2": {
            e: i
            for i, e in enumerate(
                [
                    (39, " "),
                    (40, " "),
                    (41, " "),
                    (42, " "),
                    (43, " "),
                    (44, " "),
                    (45, " "),
                    (46, " "),
                    (47, " "),
                    (48, " "),
                    (49, " "),
                    (50, " "),
                    (51, " "),
                    (52, " "),
                    (53, " "),
                    (54, " "),
                    (55, " "),
                ],
                start=1,
            )
        },  # len 17, no dels
        "h_fwr3": {
            e: i
            for i, e in enumerate(
                [
                    (66, " "),
                    (67, " "),
                    (68, " "),
                    (69, " "),
                    (70, " "),
                    (71, " "),
                    (72, " "),
                    (74, " "),
                    (75, " "),
                    (76, " "),
                    (77, " "),
                    (78, " "),
                    (79, " "),
                    (80, " "),
                    (81, " "),
                    (82, " "),
                    (83, " "),
                    (84, " "),
                    (85, " "),
                    (86, " "),
                    (87, " "),
                    (88, " "),
                    (89, " "),
                    (90, " "),
                    (91, " "),
                    (92, " "),
                    (93, " "),
                    (94, " "),
                    (95, " "),
                    (96, " "),
                    (97, " "),
                    (98, " "),
                    (99, " "),
                    (100, " "),
                    (101, " "),
                    (102, " "),
                    (103, " "),
                    (104, " "),
                ],
                start=1,
            )
        },  # len 38, del @ imgt73
        "h_fwr4": {
            e: i
            for i, e in enumerate(
                [
                    (118, " "),
                    (119, " "),
                    (120, " "),
                    (121, " "),
                    (122, " "),
                    (123, " "),
                    (124, " "),
                    (125, " "),
                    (126, " "),
                    (127, " "),
                    (128, " "),
                ],
                start=1,
            )
        },  # len 11, no dels
    },
    "l": {
        "l_cdr1": {
            e: i
            for i, e in enumerate(
                [
                    (27, " "),
                    (28, " "),
                    (29, " "),
                    (30, " "),
                    (31, " "),
                    (32, " "),
                    (32, "A"),
                    (32, "B"),
                    (32, "C"),
                    (32, "D"),
                    (33, "D"),
                    (33, "C"),
                    (33, "B"),
                    (33, "A"),
                    (33, " "),
                    (34, " "),
                    (35, " "),
                    (36, " "),
                    (37, " "),
                    (38, " "),
                ],
                start=1,
            )
        },
        "l_cdr2": {
            e: i
            for i, e in enumerate(
                [
                    (56, " "),
                    (57, " "),
                    (58, " "),
                    (59, " "),
                    (60, " "),
                    (60, "A"),
                    (60, "B"),
                    (60, "C"),
                    (61, "C"),
                    (61, "B"),
                    (61, "A"),
                    (61, " "),
                    (62, " "),
                    (63, " "),
                    (64, " "),
                    (65, " "),
                ],
                start=1,
            )
        },
        "l_cdr3": {
            e: i
            for i, e in enumerate(
                [
                    (105, " "),
                    (106, " "),
                    (107, " "),
                    (108, " "),
                    (109, " "),
                    (110, " "),
                    (111, " "),
                    (111, "A"),
                    (111, "B"),
                    (111, "C"),
                    (111, "D"),
                    (111, "E"),
                    (111, "F"),
                    (111, "G"),
                    (112, "G"),
                    (112, "F"),
                    (112, "E"),
                    (112, "D"),
                    (112, "C"),
                    (112, "B"),
                    (112, "A"),
                    (112, " "),
                    (113, " "),
                    (114, " "),
                    (115, " "),
                    (116, " "),
                    (117, " "),
                ],
                start=1,
            )
        },
        "l_fwr1": {
            e: i
            for i, e in enumerate(
                [
                    (1, " "),
                    (2, " "),
                    (3, " "),
                    (4, " "),
                    (5, " "),
                    (6, " "),
                    (7, " "),
                    (8, " "),
                    (9, " "),
                    (10, " "),
                    (11, " "),
                    (12, " "),
                    (13, " "),
                    (14, " "),
                    (15, " "),
                    (16, " "),
                    (17, " "),
                    (18, " "),
                    (19, " "),
                    (20, " "),
                    (21, " "),
                    (22, " "),
                    (23, " "),
                    (24, " "),
                    (25, " "),
                    (26, " "),
                ],
                start=1,
            )
        },  # len 26, no dels (NB this is the problem one with a possible del at 10 if locus is lambda)
        "l_fwr2": {
            e: i
            for i, e in enumerate(
                [
                    (39, " "),
                    (40, " "),
                    (41, " "),
                    (42, " "),
                    (43, " "),
                    (44, " "),
                    (45, " "),
                    (46, " "),
                    (47, " "),
                    (48, " "),
                    (49, " "),
                    (50, " "),
                    (51, " "),
                    (52, " "),
                    (53, " "),
                    (54, " "),
                    (55, " "),
                ],
                start=1,
            )
        },  # len 17, no dels
        "l_fwr3": {
            e: i
            for i, e in enumerate(
                [
                    (66, " "),
                    (67, " "),
                    (68, " "),
                    (69, " "),
                    (70, " "),
                    (71, " "),
                    (72, " "),
                    (74, " "),
                    (75, " "),
                    (76, " "),
                    (77, " "),
                    (78, " "),
                    (79, " "),
                    (80, " "),
                    (83, " "),
                    (84, " "),
                    (85, " "),
                    (86, " "),
                    (87, " "),
                    (88, " "),
                    (89, " "),
                    (90, " "),
                    (91, " "),
                    (92, " "),
                    (93, " "),
                    (94, " "),
                    (95, " "),
                    (96, " "),
                    (97, " "),
                    (98, " "),
                    (99, " "),
                    (100, " "),
                    (101, " "),
                    (102, " "),
                    (103, " "),
                    (104, " "),
                ],
                start=1,
            )
        },  # len 36, del imgt73,81,82
        "l_fwr4": {
            e: i
            for i, e in enumerate(
                [
                    (118, " "),
                    (119, " "),
                    (120, " "),
                    (121, " "),
                    (122, " "),
                    (123, " "),
                    (124, " "),
                    (125, " "),
                    (126, " "),
                    (127, " "),
                ],
                start=1,
            )
        },  # len 10, no dels (ends 127)
    },
}
