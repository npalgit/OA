package commons;

/**
 * Created by tao on 9/10/17.
 */
public class SortAlgorithms {

    public void countSort(int []nums){
        int maxVal=Integer.MIN_VALUE;
        int minVal=Integer.MAX_VALUE;
        for(int x:nums){
            maxVal=Math.max(maxVal,x);
            minVal=Math.min(minVal,x);
        }
        int[]count=new int[maxVal-minVal+1];
        int n=count.length;
        for(int x:nums){
            count[x-minVal]++;
        }
        for(int i=1;i<n;++i)
            count[i]+=count[i-1];
        int nn=nums.length;
        int[]copy=nums.clone();
        for(int i=nn-1;i>=0;--i){
            nums[count[copy[i]-minVal]-1]=copy[i];
            count[copy[i]-minVal]--;
        }

    }
}
