import re
def filterSRC(src):
    srclist = re.findall(r'[(](.*?)[)]', src)
    predicatelist=[]
    problist=[]
    for item in srclist:
        item = item.replace('(','')
        item = item.replace(')','')
        predicatelist.append(item.split(':')[0])
        problist.append(item.split(':')[1])
    return predicatelist,problist

import openpyxl
source = "(on:0.6929) (has:0.6739) (wearing:0.8970) (of:0.6764) (in:0.5022) (near:0.3078) (behind:0.4551) (with:0.4158) (holding:0.7244) (above:0.4810) (sitting on:0.7244) (wears:0.8707) (under:0.5668) (riding:0.8977) (in front of:0.4139) (standing on:0.7430) (at:0.7679) (carrying:0.8302) (attached to:0.6165) (walking on:0.9196) (over:0.5599) (for:0.5911) (looking at:0.6916) (watching:0.6948) (hanging from:0.6538) (laying on:0.8405) (eating:0.8539) (and:0.5281) (belonging to:0.8658) (parked on:0.9050) (using:0.8550) (covering:0.7076) (between:0.4236) (along:0.6896) (covered in:0.7020) (part of:0.6986) (lying on:0.7787) (on back of:0.7908) (to:0.6974) (walking in:0.8122) (mounted on:0.7708) (across:0.5780) (against:0.6210) (from:0.6498) (growing on:0.7543) (painted on:0.7265) (playing:0.6212) (made of:0.5938) (says:0.7500) (flying in:0.0000) "
predicatelist,problist_mR_VSL = filterSRC(source)
for i in problist_mR_VSL:
    print(i)

mybook=openpyxl.load_workbook("/home/share/zhanghao/data/image/datasets/output/Group/boxaug/v10.xlsx",data_only=True)
mySheet=mybook.active

#按行获取新书表的单元格（第一行除外--标题，不是数据）
myRows=list(mySheet.values)[1:]

mydics={}
for myRow in myRows:
    mydics[myRow[0]]=myRow[1]

# mybook.save("结果表.xlsx")
cls_num = []
num = 0
for key in predicatelist:
    # cls_num.append(mydics[key])
   print(mydics[key])
print(num)
print(predicatelist)




source = "(on:0.1192) (has:0.2434) (wearing:0.0242) (of:0.0463) (in:0.0694) (near:0.0511) (behind:0.1119) (with:0.0216) (holding:0.0538) (above:0.0301) (sitting on:0.0894) (wears:0.3623) (under:0.1709) (riding:0.2961) (in front of:0.2105) (standing on:0.1341) (at:0.3076) (carrying:0.3377) (attached to:0.0288) (walking on:0.1865) (over:0.0366) (for:0.0926) (looking at:0.0435) (watching:0.0784) (hanging from:0.2831) (laying on:0.1429) (eating:0.4048) (and:0.1452) (belonging to:0.1857) (parked on:0.5325) (using:0.0000) (covering:0.1952) (between:0.0000) (along:0.2051) (covered in:0.2143) (part of:0.0469) (lying on:0.1556) (on back of:0.1364) (to:0.2778) (walking in:0.0000) (mounted on:0.0435) (across:0.0556) (against:0.0526) (from:0.0000) (growing on:0.0000) (painted on:0.1429) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)"
predicatelist,problist_mR_VSL = filterSRC(source)
for i in problist_mR_VSL:
    print(i)

a = 0
# mR_VSL = '(on:0.0417) (has:0.1524) (wearing:0.1109) (of:0.1360) (in:0.1370) (near:0.0471) (behind:0.3671) (with:0.2636) (holding:0.2966) (above:0.1484) (sitting on:0.4407) (wears:0.7291) (under:0.3497) (riding:0.6923) (in front of:0.2669) (standing on:0.2314) (at:0.6127) (carrying:0.6541) (attached to:0.2196) (walking on:0.6927) (over:0.3450) (for:0.4195) (looking at:0.2859) (watching:0.5383) (hanging from:0.4963) (laying on:0.6137) (eating:0.7640) (and:0.5316) (belonging to:0.5656) (parked on:0.9298) (using:0.5912) (covering:0.6447) (between:0.2639) (along:0.5306) (covered in:0.6770) (part of:0.2160) (lying on:0.1020) (on back of:0.5083) (to:0.5137) (walking in:0.2852) (mounted on:0.2024) (across:0.2103) (against:0.2742) (from:0.3042) (growing on:0.3685) (painted on:0.3994) (playing:0.2955) (made of:0.2812) (says:0.3333) (flying in:0.0000) '
# predicatelist,problist_mR_VSL = filterSRC(mR_VSL)
# for i in problist_mR_VSL:
#     print(i)
# print("**************************")
# mR_VS = '(on:0.0667) (has:0.2120) (wearing:0.1427) (of:0.1626) (in:0.1597) (near:0.0719) (behind:0.4045) (with:0.2793) (holding:0.3328) (above:0.1678) (sitting on:0.4438) (wears:0.6970) (under:0.3914) (riding:0.6880) (in front of:0.2677) (standing on:0.2673) (at:0.6273) (carrying:0.6607) (attached to:0.2615) (walking on:0.6930) (over:0.3361) (for:0.4268) (looking at:0.2206) (watching:0.5387) (hanging from:0.5016) (laying on:0.5608) (eating:0.7777) (and:0.5173) (belonging to:0.5472) (parked on:0.9298) (using:0.5354) (covering:0.6210) (between:0.2326) (along:0.5031) (covered in:0.6389) (part of:0.2027) (lying on:0.1173) (on back of:0.4243) (to:0.5164) (walking in:0.1972) (mounted on:0.1722) (across:0.2738) (against:0.2177) (from:0.2732) (growing on:0.2953) (painted on:0.4392) (playing:0.1477) (made of:0.2812) (says:0.2500) (flying in:0.0000) '
# _,problist_mR_VS = filterSRC(mR_VS)
# for i in problist_mR_VS:
#     print(i)
# print("**************************")
# mR_SL = "(on:0.1133) (has:0.3392) (wearing:0.3141) (of:0.3098) (in:0.2390) (near:0.0871) (behind:0.4236) (with:0.2946) (holding:0.3130) (above:0.1547) (sitting on:0.4074) (wears:0.5791) (under:0.3757) (riding:0.6673) (in front of:0.2717) (standing on:0.2792) (at:0.5878) (carrying:0.6635) (attached to:0.2580) (walking on:0.7832) (over:0.3576) (for:0.4360) (looking at:0.2455) (watching:0.5146) (hanging from:0.5574) (laying on:0.5395) (eating:0.7684) (and:0.4196) (belonging to:0.5105) (parked on:0.9003) (using:0.4937) (covering:0.5894) (between:0.1854) (along:0.4587) (covered in:0.5020) (part of:0.1245) (lying on:0.0485) (on back of:0.3794) (to:0.4836) (walking in:0.0493) (mounted on:0.0632) (across:0.1825) (against:0.1532) (from:0.2282) (growing on:0.1810) (painted on:0.2707) (playing:0.0455) (made of:0.2812) (says:0.0833) (flying in:0.0000) "
# _,problist_mR_SL = filterSRC(mR_SL)
# for i in problist_mR_SL:
#     print(i)
# print("**************************")
#
# maxlist = []
# for i in range(len(predicatelist)):
#    maxlist.append(max(problist_mR_VSL[i],problist_mR_SL[i],problist_mR_VS[i]))
# maxlist_num = []
# for i in maxlist:
#     maxlist_num.append(float(i))
# print("mR：")
# print(sum(maxlist_num)/50)
#
# cls_num_list = [63023, 25140, 20148, 19501, 11383, 7394, 4558, 5358, 4657, 2388, 2300, 2324, 1705, 1729, 1276, 1409, 749, 687, 838, 648, 491, 354, 349, 392, 407, 334, 338, 216, 743, 292, 221, 243, 216, 155, 196, 142, 170, 178, 153, 113, 169, 82, 69, 98, 101, 102, 29, 37, 12, 25]
# tmpSum = 0
# for i in range(len(predicatelist)):
#     tmpSum += cls_num_list[i]*maxlist_num[i]
# print("R:")
# print(tmpSum/sum(cls_num_list))
