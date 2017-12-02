import commons.TreeNode;

import java.util.Stack;

public class ForUSAll {



    public int largestCompleteTree(TreeNode root,int[]ans){
        if(root==null)
            return 0;
        if(root.left==null && root.right==null)
            return 1;
        int l = largestCompleteTree(root.left,ans);
        int r = largestCompleteTree(root.right,ans);
        ans[0]=Math.max(ans[0],Math.min(l,r)+1);
        return Math.min(l,r)+1;
    }
    public int largestCompleteTree(TreeNode root){
        int []ans ={0};
        largestCompleteTree(root,ans);
        return (1<<ans[0])-1;
    }





    public int getMax(int[]nums,int begin,int end,int n){
        //get n numbers, and make its maximum
        int ans = 0,start=begin;
        int sum=0;
        for(int i=begin;i<=end;++i){
            sum+=nums[i];
            if(i-start>n-1)
                sum-=nums[start++];
            ans=Math.max(ans,sum);

        }
        return ans;
    }

    public boolean greater(int[]nums1,int start1,int[]nums2,int start2){
        int m =nums1.length,n=nums2.length;
        while(start1<m && start2<n){
            if(nums1[start1]==nums2[start2]){
                start1++;
                start2++;
            }else if(nums1[start1]<nums2[start2])
                return false;
            else
                return true;
        }
        return start1!=m;
    }

    public int solution(int[] movies, int K, int L) {
        // write your code in Java SE 8
        //get k movies from one part of array movies
        //get another L moveis from another part
        //maximum it

        //enumerate all endponts that setparate the movies into two part;

        int n = movies.length;
        if(K+L>n)
            return -1;
        int ans =0;
        for(int endpoint = K-1;endpoint<n-L;++endpoint){
            int one = getMax(movies,0,endpoint,K);
            int two = getMax(movies,endpoint+1,n-1,L);
            ans=Math.max(one+two,ans);

        }
        return ans;
    }
}
