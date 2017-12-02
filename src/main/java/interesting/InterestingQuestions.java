package interesting;

import java.util.*;

/**
 * Created by tao on 9/16/17.
 */
public class InterestingQuestions {

    public int maxSumNoLargerThanK(int []nums,int k){
        int sum =0,maxVal=Integer.MIN_VALUE;
        TreeSet<Integer> set =new TreeSet<>();
        set.add(0);
        for(int x:nums){
            sum +=x;
            Integer it = set.ceiling(sum-k);
            if(it!=null)
                maxVal=Math.max(maxVal,sum-it);
            set.add(sum);
        }
        return maxVal;
    }

    //if all number is positive , you can use two pointers
    public int maxSum(int[]nums,int target){
        int n=nums.length,begin=0,end=0,sum=0,maxVal=0;
        while(end<n){
            sum+=nums[end++];
            if(sum==target)
                return target;
            while(sum>target){
                sum-=nums[begin++];
            }
            maxVal=Math.max(maxVal,sum);
        }
        return maxVal;
    }

    //merge two sorted array to get maximum number

    public boolean greater(int[]nums1,int start1,int[]nums2,int start2){
        int m = nums1.length,n = nums2.length;
        while(start1<m && start2<n){
            if(nums1[start1]>nums2[start2])
                return true;
            else if(nums1[start1]<nums2[start2])
                return false;
            else{
                start1++;
                start2++;
            }
        }
        return start1!=m;

    }
    public int []getMaxi(int[]nums1,int[]nums2){
        int m=  nums1.length,n=nums2.length;
        int k = m+n;
        int []res=new int[k];
        int ind =0,start1=0,start2=0;
        while(ind<k){
            res[ind++]=greater(nums1,start1,nums2,start2)?nums1[start1++]:nums2[start2++];
        }
        return res;
    }

    public int odd(int n){
        if(n<=2)
            return 1;
        return ((n&0x1)==0?2*(lastRemaining(n/2))-1:2*lastRemaining(n/2));
    }
    public int lastRemaining(int n) {
        if(n<=2)
            return n;
        return 2*((n&0x1)==0?odd(n/2):odd((n-1)/2));
    }

    //424 Longest Repeating Character Replacement
    //很有意思的地方在于不需要更新max，其实就是自动把<len 的string过滤了
    public int characterReplacement(String s, int k) {
        char []ss=s.toCharArray();
        int n = ss.length,start=0,end=0,len=0,maxCnt=0;
        int []cnt = new int[26];
        while(end<n){
            maxCnt=Math.max(maxCnt,++cnt[ss[end++]-'A']);
            while(end-start-maxCnt>k){
                cnt[ss[start++]-'A']--;
            }
            len = Math.max(len,end-start);
        }
        return len;
    }


    Map<Integer,Boolean> map=new HashMap<>();
    public boolean canWin(int maxVal,int desired,int used){
        if(desired<=0){
            map.put(used,false);
            return false;
        }
        if(map.containsKey(used))
            return map.get(used);
        for(int i=1;i<=maxVal;++i){

            //doesnot visited;
            if(((used>>i)&0x1)==0){
                used|=(1<<i);
                boolean next = !canWin(maxVal,desired-i,used);
                used-=(1<<i);
                if(next){
                    map.put(used,true);
                    return true;
                }
            }
        }
        map.put(used,false);
        return false;
    }
    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        //linkedin 一道题，很有意思，需要用到bit
        int sum = (maxChoosableInteger+1)*maxChoosableInteger/2;
        if(sum<desiredTotal)
            return  false;
        return canWin(maxChoosableInteger,desiredTotal,0);
    }




}
