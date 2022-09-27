import math
import numpy as np

LIMB_INFO = dict(
    fourdag_19=dict(
        n_kps=19,
        n_pafs=18,
        shape_size=10,
        joint_parent=[
            -1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 11, 12, 14, 13
        ],
        hierarchy_map=[
            0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3
        ],
        paf_dict=[[1, 2, 7, 0, 0, 3, 8, 1, 5, 11, 5, 1, 6, 12, 6, 1, 14, 13],
                  [
                      0, 7, 13, 2, 3, 8, 14, 5, 11, 15, 9, 6, 12, 16, 10, 4,
                      17, 18
                  ]],
        m_joints=[
            0.0018322300165891647, -0.00010963100066874176,
            -0.1023780032992363, 0.11452200263738632, 0.027707800269126892,
            -0.1321370005607605, 0.13699699938297272, -0.10021399706602097,
            0.12022099643945694, -0.013417200185358524, 0.03774090111255646,
            -0.40522998571395874, 0.399260014295578, -0.10806600004434586,
            0.08671879768371582, -0.6374769806861877, 0.6368650197982788,
            0.11188799887895584, -0.12464399635791779, -0.43331000208854675,
            0.09824070334434509, -0.4318639934062958, -0.42760899662971497,
            0.42442700266838074, 0.09882839769124985, 0.10586100071668625,
            -0.8320649862289429, -0.8374119997024536, 0.2976999878883362,
            0.3440679907798767, 0.06376519799232483, 0.038992200046777725,
            -1.2320200204849243, -1.2276500463485718, 0.05516450107097626,
            0.04539240151643753, -1.2767900228500366, -1.2848199605941772,
            0.006065589841455221, -0.030888399109244347, 0.00878852978348732,
            0.03920949995517731, 0.13028299808502197, -0.019071299582719803,
            -0.007281539961695671, 0.003349659964442253, 0.0059417798183858395,
            0.04202660173177719, 0.007522290106862783, -0.04162359982728958,
            -0.04715130105614662, -0.036375198513269424, -0.04684149846434593,
            -0.05510310083627701, -0.05260920152068138, 0.08009690046310425,
            0.08515729755163193
        ],
        shape_blend=[
            -0.0003980599867645651,
            -0.00011868900037370622,
            -2.9507400540751405e-05,
            0.000222454997128807,
            6.802160351071507e-05,
            0.0001191140036098659,
            1.7645899788476527e-05,
            -7.789880100972368e-07,
            9.732579928822815e-05,
            1.572700057295151e-05,
            -0.024616699665784836,
            0.0008205170161090791,
            -0.012188499793410301,
            0.0031967998947948217,
            0.0021081999875605106,
            0.005194710101932287,
            -0.0013632499612867832,
            -0.0007316800183616579,
            -0.0025655198842287064,
            -0.0001380119938403368,
            0.003493440104648471,
            0.007923279888927937,
            0.003416369901970029,
            -0.00026210900978185236,
            -0.0025999799836426973,
            0.003702230053022504,
            -5.5152700952021405e-05,
            0.0005877729854546487,
            0.0010699500562623143,
            -0.0005388340214267373,
            0.00011704100325005129,
            0.000399323005694896,
            0.00037406399496831,
            0.00020244800543878227,
            4.099799843970686e-05,
            0.00010774400288937613,
            -5.159629836271051e-06,
            6.446050247177482e-05,
            7.817889854777604e-05,
            4.576840001391247e-05,
            0.008318549953401089,
            -0.0007871040143072605,
            -0.0006884189788252115,
            0.0010070500429719687,
            -0.0029316600412130356,
            -0.0019596500787883997,
            -0.0006023889873176813,
            -0.0003421340079512447,
            0.00017485000716987997,
            2.9364000511122867e-05,
            -0.003121060086414218,
            -0.0017387500265613198,
            -3.440630052864435e-06,
            0.001419579959474504,
            0.0005531240021809936,
            -2.0717499864986166e-05,
            0.0017533099744468927,
            0.0005213890108279884,
            7.267959881573915e-05,
            -8.125029853545129e-05,
            -0.0034143798984587193,
            -0.004969520028680563,
            -0.004051229916512966,
            -0.003643119940534234,
            -0.0018645100062713027,
            -0.000556221988517791,
            -0.0010985699482262135,
            0.0010963899549096823,
            -0.0008559490088373423,
            0.0003273439942859113,
            -0.025414299219846725,
            0.0017124799778684974,
            -0.01106529962271452,
            0.0040114400908350945,
            0.0026048000436276197,
            0.00562551012262702,
            -0.001098550041206181,
            -0.0006787360180169344,
            -0.0023665500339120626,
            -0.00013644200225826353,
            0.0033887699246406555,
            0.006476960144937038,
            0.002724830061197281,
            -0.0004623529966920614,
            -0.0025292099453508854,
            0.0032830899581313133,
            -0.0001360729947919026,
            0.0001633159990888089,
            0.0011753699509426951,
            -0.0004619770043063909,
            0.0033724899403750896,
            0.00476772990077734,
            0.0035651300568133593,
            0.003727490082383156,
            0.0019821799360215664,
            0.000496807973831892,
            0.0012299499940127134,
            -0.0011482399422675371,
            0.0008286800002679229,
            -0.0004504910029936582,
            -0.02440039999783039,
            0.0029490399174392223,
            -0.011094599962234497,
            0.0034197100903838873,
            0.0018023400334641337,
            0.004188249818980694,
            -0.0013911599526181817,
            -0.0010619900422170758,
            -0.0028982500080019236,
            -0.0004898710176348686,
            0.003137570107355714,
            0.008400609716773033,
            0.004671869799494743,
            0.0017864999826997519,
            -0.0015993199776858091,
            0.0036661599297076464,
            0.001153170014731586,
            0.00047433399595320225,
            0.0012803099816665053,
            -4.029889896628447e-05,
            0.002605050103738904,
            0.000889630988240242,
            -0.000757434987463057,
            -0.0013561700470745564,
            -0.00042205199133604765,
            -0.0012696300400421023,
            -0.0003871950029861182,
            -0.0005131629877723753,
            -0.0008624609909020364,
            0.0004142480029258877,
            0.02172520011663437,
            -0.007389550097286701,
            0.005084650125354528,
            -0.00596485985442996,
            0.004890790209174156,
            -6.788589962525293e-05,
            0.0006118100136518478,
            -0.0018875099485740066,
            -0.0006803990108892322,
            0.000265520007815212,
            0.0029029399156570435,
            0.0038713798858225346,
            0.005503830034285784,
            0.006421309895813465,
            0.006133500020951033,
            0.0006375340162776411,
            -0.0010648899478837848,
            0.0009631579741835594,
            0.0013433099957183003,
            -0.001259359996765852,
            -0.008929899893701077,
            -0.007059189956635237,
            -0.002211519982665777,
            -6.965579814277589e-05,
            -0.0007484150119125843,
            0.0016686900053173304,
            -0.0011512499768286943,
            6.507999933091924e-05,
            0.0015289300354197621,
            0.0005656849825754762,
            0.007305010221898556,
            -0.0001517930068075657,
            -0.0009416589746251702,
            0.0024064600002020597,
            -0.0023471100721508265,
            -0.0013090299908071756,
            -0.000240408000536263,
            -0.00015356299991253763,
            0.0004080829967278987,
            0.00016870300169102848,
            -0.0017817800398916006,
            -0.0008903879788704216,
            2.8494299840531312e-05,
            0.001023740042001009,
            0.00012866800534538925,
            2.2065099983592518e-05,
            0.0020449000876396894,
            1.765510023687966e-05,
            -0.0001494979951530695,
            -9.271760063711554e-05,
            0.009656240232288837,
            0.007572989910840988,
            0.0022630998864769936,
            -9.708030120236799e-05,
            0.0007323200115934014,
            -0.0018285400001332164,
            0.0012104000197723508,
            1.0423300409456715e-06,
            -0.0016119299689307809,
            -0.0005791970179416239,
            0.008099040016531944,
            -0.000352992006810382,
            -0.0009666130063124001,
            0.0018469099886715412,
            -0.002690309891477227,
            -0.002020779997110367,
            -0.0004566280113067478,
            -0.0004074170137755573,
            0.00027066501206718385,
            3.2586598535999656e-05,
            -0.0016034099971875548,
            0.0009043680038303137,
            0.0013213000493124127,
            0.0021273500751703978,
            0.00073273602174595,
            0.00011440899834269658,
            0.002535539912059903,
            0.0003652909945230931,
            -0.0005137149710208178,
            -8.914220234146342e-05,
            -0.004572560079395771,
            -0.003391799982637167,
            -0.002324290107935667,
            -0.001936870045028627,
            -0.0011773400474339724,
            -6.515799759654328e-05,
            -0.00048803098616190255,
            0.0028196400962769985,
            0.0002361369988648221,
            0.0005132199730724096,
            -0.04774609953165054,
            0.009952659718692303,
            -0.00686269998550415,
            -0.0008507610182277858,
            0.002748809987679124,
            0.0020530300680547953,
            -0.00045581901213154197,
            0.00034346801112405956,
            -0.0005917859962210059,
            0.0003055789857171476,
            0.0061587500385940075,
            0.01072010025382042,
            0.0036963699385523796,
            -0.002062849933281541,
            -0.004079720005393028,
            0.002863609930500388,
            -0.0013847900554537773,
            -0.001309809973463416,
            0.00036883598659187555,
            -0.00036910499329678714,
            0.00491386977955699,
            0.0035370199475437403,
            0.0023562400601804256,
            0.0024174400605261326,
            0.0015287300338968635,
            -9.203409717883915e-05,
            0.0006551520200446248,
            -0.003599239978939295,
            -2.397470098003396e-06,
            -0.0006820959970355034,
            -0.04842640087008476,
            0.00975123979151249,
            -0.006704510189592838,
            -0.0008572409860789776,
            0.0028836498968303204,
            0.001893150038085878,
            -0.00039050300256349146,
            0.0001877839968074113,
            -0.00043747900053858757,
            0.00030441099079325795,
            0.006107610184699297,
            0.0109393997117877,
            0.0039602299220860004,
            -0.001810119953006506,
            -0.003931600134819746,
            0.002965020015835762,
            -0.001271620043553412,
            -0.0012725299457088113,
            0.0003731460019480437,
            -0.0003911030071321875,
            -0.0005785020184703171,
            -0.002860599895939231,
            -0.002288700081408024,
            -0.001313620014116168,
            -0.0004884289810433984,
            -0.0003980690089520067,
            -0.000536305014975369,
            -0.00029132800409570336,
            -7.420349902531598e-06,
            0.00017987699538934976,
            0.018786299973726273,
            -0.0018859199481084943,
            0.002615999896079302,
            -0.002592789940536022,
            0.0009399399859830737,
            -0.0011024200357496738,
            -0.0006407559849321842,
            -0.0010344400070607662,
            -0.001230970025062561,
            1.8775399439618923e-05,
            0.0012571399565786123,
            0.005489380098879337,
            0.0047562001273036,
            0.004460370168089867,
            0.0035553500056266785,
            -0.001656329957768321,
            0.0006094489945098758,
            0.0006979770259931684,
            -0.0011129500344395638,
            -0.0010059199994429946,
            0.0025828599464148283,
            0.0021015799138695,
            0.00036597400321625173,
            -0.0008808069978840649,
            -0.00011360400094417855,
            -0.0008938509854488075,
            4.7305900807259604e-05,
            0.00030974700348451734,
            -0.0009915060363709927,
            0.00021129999367985874,
            0.019003000110387802,
            -0.0027095701079815626,
            0.00472267996519804,
            -0.0019649099558591843,
            0.0031912000849843025,
            0.002225029980763793,
            0.0003873840032611042,
            -0.00029578700195997953,
            -0.0001506120024714619,
            0.0010308800265192986,
            -0.00048737498582340777,
            -0.0027587199583649635,
            -0.0002579930005595088,
            0.0015516499988734722,
            0.0026714899577200413,
            -0.0007549090078100562,
            -0.0021907601039856672,
            0.0005574999959208071,
            0.0005732899880968034,
            -0.0014788300031796098,
            -0.025372900068759918,
            -0.008133579976856709,
            0.0019022299675270915,
            0.0003251100133638829,
            -0.0016716100508347154,
            0.0017389700515195727,
            -0.0018183400388807058,
            -0.00036380800884217024,
            0.0009857160039246082,
            0.0009852079674601555,
            0.005243400111794472,
            0.0009496320271864533,
            -0.0003685469855554402,
            0.0027647700626403093,
            -0.0031297500245273113,
            -0.0019086400279775262,
            -0.00018181400082539767,
            -5.808709829580039e-06,
            0.00017024099361151457,
            -0.00010663799912435934,
            -0.003788999980315566,
            -0.002727830084040761,
            -0.0008885330171324313,
            0.0005337530164979398,
            -0.00048052301281131804,
            0.00046323699643835425,
            0.003044690005481243,
            7.697610271861777e-05,
            7.043660298222676e-05,
            0.00025902900961227715,
            0.024809500202536583,
            0.00858255010098219,
            -0.0015187800163403153,
            -0.00030673699802719057,
            0.0017185399774461985,
            -0.0018828300526365638,
            0.0018714099423959851,
            0.0002965769963338971,
            -0.0012034999672323465,
            -0.0009929010411724448,
            0.0038920000661164522,
            -0.0001375010033370927,
            -0.0012356899678707123,
            0.0022429500240832567,
            -0.003561390098184347,
            -0.0021001799032092094,
            -0.0003059319860767573,
            0.00016720499843358994,
            0.0002528360055293888,
            -0.00020823099475819618,
            -0.004383780062198639,
            -0.0026290901005268097,
            -0.0006730729946866632,
            0.0008682699990458786,
            -0.00034221200621686876,
            0.00046641999506391585,
            0.003322170116007328,
            0.00017894800112117082,
            1.585340032761451e-05,
            0.0002712190034799278,
            -0.0054176198318600655,
            -0.004132559988647699,
            -0.002586370101198554,
            -0.00230022007599473,
            -0.0010289299534633756,
            -0.0002710930129978806,
            -0.0005706200026907027,
            0.0033033699728548527,
            0.0006192880100570619,
            0.0010404599597677588,
            -0.07361090183258057,
            0.013650099746882915,
            -0.00183958001434803,
            -0.005292980000376701,
            0.0027403701096773148,
            -0.0028952599968761206,
            0.0011280899634584785,
            -6.257189670577645e-05,
            0.0012069999938830733,
            -4.110169902560301e-05,
            0.00508992001414299,
            0.010249299928545952,
            0.001882239943370223,
            -0.00494761997833848,
            -0.005239080172032118,
            0.0024449098855257034,
            -0.0026042000390589237,
            -0.0007306329789571464,
            -0.00035462601226754487,
            0.0003490149974822998,
            0.0038521999958902597,
            0.002898520091548562,
            0.0016802400350570679,
            0.002108759945258498,
            0.0008423550170846283,
            0.00036115600960329175,
            0.0006169269909150898,
            -0.0032198100816458464,
            -0.0006774530047550797,
            -0.001192330033518374,
            -0.07337780296802521,
            0.013252399861812592,
            -0.0020734600257128477,
            -0.005442500114440918,
            0.002847020048648119,
            -0.0028209600131958723,
            0.0011297500459477305,
            -0.0001381470065098256,
            0.0011802399531006813,
            1.9151200831402093e-05,
            0.004397119861096144,
            0.009995129890739918,
            0.0017618200508877635,
            -0.00502733001485467,
            -0.005270869936794043,
            0.002482079900801182,
            -0.002620500046759844,
            -0.000656360003631562,
            -0.00033956998959183693,
            0.00022616400383412838,
            -0.04017990082502365,
            -0.007416329812258482,
            0.006222870200872421,
            0.0006003309972584248,
            -0.0016779700526967645,
            0.0006593589787371457,
            -0.0012731000315397978,
            -0.00020251600653864443,
            -0.00039522998849861324,
            0.0005100290291011333,
            0.004392129834741354,
            0.001213619951158762,
            -0.0003690799931064248,
            0.003542599966749549,
            -0.003240989986807108,
            -0.0021059399005025625,
            -0.00017557099636178464,
            0.0002499129914212972,
            0.00018284299585502595,
            -3.174980156472884e-05,
            -0.0029378400649875402,
            -0.0013614499475806952,
            -0.00042965300963260233,
            -7.711369835305959e-05,
            -0.0009232269949279726,
            -7.676750101381913e-05,
            0.0034673199988901615,
            -0.0002794039901345968,
            -0.0008617640123702586,
            0.0004173800116404891,
            0.04002479836344719,
            0.0074407500214874744,
            -0.006158050149679184,
            -0.0005538969999179244,
            0.0017026199493557215,
            -0.0006783109856769443,
            0.0012727200519293547,
            8.884150156518444e-05,
            0.00038900598883628845,
            -0.0005135260289534926,
            0.004239359870553017,
            0.0013997299829497933,
            -0.0005783929955214262,
            0.003326730104163289,
            -0.0034160299692302942,
            -0.002259189961478114,
            -0.0002851790050044656,
            0.00028374200337566435,
            4.00140015699435e-05,
            -9.97200986603275e-05,
            -0.002900729887187481,
            -0.001056330045685172,
            -0.000246460986090824,
            0.00017260400636587292,
            -0.000852264987770468,
            -3.762829919651267e-06,
            0.0035263500176370144,
            -0.0002126420004060492,
            -0.0008252250263467431,
            0.00046142301289364696,
            0.0061552999541163445,
            0.003786920104175806,
            0.0022906900849193335,
            0.0015395799418911338,
            0.0008029550081118941,
            0.00017010699957609177,
            8.506079757353291e-05,
            -0.00322604994289577,
            -0.0008119390113279223,
            -0.0007143549737520516,
            -0.07826510071754456,
            0.012709399685263634,
            -0.00010163499973714352,
            -0.006216479931026697,
            0.00385512993671,
            -0.002179390052333474,
            0.0020661999005824327,
            0.00011030300083803013,
            0.002017199993133545,
            -2.6802099455380812e-05,
            0.013299800455570221,
            0.007371710147708654,
            -0.0011215700069442391,
            -0.007770949974656105,
            -0.006386950146406889,
            0.0026745200157165527,
            -0.004695559851825237,
            -0.001145330024883151,
            0.00015606099623255432,
            9.197409963235259e-05,
            -0.006221740040928125,
            -0.003200510051101446,
            -0.0016708699986338615,
            -0.0013594599440693855,
            -0.0007932899752631783,
            -0.00013672899513039738,
            -0.00014413800090551376,
            0.003169979900121689,
            0.0008632470271550119,
            0.0008664419874548912,
            -0.07853379845619202,
            0.013753499835729599,
            0.0004113389877602458,
            -0.005927000194787979,
            0.0038871699944138527,
            -0.0019264599541202188,
            0.00212032999843359,
            0.00029300100868567824,
            0.002053739968687296,
            0.00010213199857389554,
            0.013594700023531914,
            0.0066468301229178905,
            -0.0017055700300261378,
            -0.00810530036687851,
            -0.006610089913010597,
            0.0025961899664252996,
            -0.004910250194370747,
            -0.0012272399617359042,
            0.00019115199393127114,
            0.00016954299644567072,
        ]),
    openpose_25=dict(
        n_kps=25,
        n_pafs=26,
        hierarchy_map=[
            0, 0, 0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 1, 1, 2, 2, 3, 3, 3,
            3, 3, 3
        ],
        paf_dict=[[
            1, 9, 10, 8, 8, 12, 13, 1, 2, 3, 2, 1, 5, 6, 5, 1, 0, 0, 15, 16,
            14, 19, 14, 11, 22, 11
        ],
                  [
                      8, 10, 11, 9, 12, 13, 14, 2, 3, 4, 17, 5, 6, 7, 18, 0,
                      15, 16, 17, 18, 19, 20, 21, 22, 23, 24
                  ]]),
    coco=dict(
        n_kps=17,
        n_pafs=18,
        hierarchy_map=[0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        paf_dict=[[0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 5, 6, 11, 11, 12, 13, 14],
                  [
                      1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 12, 13, 14, 15,
                      16
                  ]],
    ),
)


def welsch(c, x):
    x = x / c
    return 1 - math.exp(-x * x / 2)


def line2linedist(pa, raya, pb, rayb):
    if abs(np.vdot(raya, rayb)) < 1e-5:
        return point2linedist(pa, pb, raya)
    else:
        ve = np.cross(raya, rayb)
        ve = ve / np.linalg.norm(ve)
        ve = abs(np.vdot((pa - pb), ve))
        return ve


def point2linedist(pa, pb, ray):
    ve = np.cross(pa - pb, ray)
    return np.linalg.norm(ve)


def skew(vec):
    m_skew = np.zeros((3, 3), dtype=np.float32)
    m_skew = np.array( [0, -vec[2], vec[1], \
        vec[2], 0, -vec[0], \
        -vec[1], vec[0], 0],dtype=np.float32).reshape((3, 3))
    return m_skew


def rodrigues(vec):
    theta = np.linalg.norm(vec)
    identity = np.identity(3, dtype=np.float32)
    if abs(theta) < 1e-5:
        return identity
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        r = vec / theta
        return c * identity + np.matmul((1 - c) * r.reshape(
            (-1, 1)), r.reshape((1, -1))) + s * skew(r)


def rodrigues_jacobi(vec):
    theta = np.linalg.norm(vec)
    d_skew = np.zeros((3, 9), dtype=np.float32)
    d_skew[0, 5] = d_skew[1, 6] = d_skew[2, 1] = -1
    d_skew[0, 7] = d_skew[1, 2] = d_skew[2, 3] = 1
    if abs(theta) < 1e-5:
        return -d_skew
    else:
        c = np.cos(theta)
        s = np.sin(theta)
        c1 = 1 - c
        itheta = 1 / theta
        r = vec / theta
        rrt = np.matmul(r.reshape((-1, 1)), r.reshape((1, -1)))
        m_skew = skew(r)
        identity = np.identity(3, dtype=np.float32)
        drrt = np.array([
            r[0] + r[0], r[1], r[2], r[1], 0, 0, r[2], 0, 0, 0, r[0], 0, r[0],
            r[1] + r[1], r[2], 0, r[2], 0, 0, 0, r[0], 0, 0, r[1], r[0], r[1],
            r[2] + r[2]
        ],
                        dtype=np.float32).reshape((3, 9))
        jaocbi = np.zeros((3, 9), dtype=np.float32)
        a = np.zeros((5, 1), dtype=np.float32)
        for i in range(3):
            a = np.array([
                -s * r[i], (s - 2 * c1 * itheta) * r[i], c1 * itheta,
                (c - s * itheta) * r[i], s * itheta
            ],
                         dtype=np.float32).reshape((5, 1))
            for j in range(3):
                for k in range(3):

                    jaocbi[i, k + k + k + j] = (
                        a[0] * identity[j, k] + a[1] * rrt[j, k] +
                        a[2] * drrt[i, j + j + j + k] + a[3] * m_skew[j, k] +
                        a[4] * d_skew[i, j + j + j + k])
        return jaocbi
