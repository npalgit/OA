import java.util.*;

/**
 * Created by tao on 10/18/17.
 */
public class Intuit {


    //build relationship


    public List<List<String>>tasksByLevel(List<List<String>>inputs){
        //topological sort

        Map<String,List<String>> adjacent = new HashMap<>();
        Map<String,Integer>indegree = new HashMap<>();
        for(List<String>str:inputs){
            if(!adjacent.containsKey(str.get(0)))
                adjacent.put(str.get(0),new ArrayList<>());
            adjacent.get(str.get(0)).add(str.get(1));
            int val = indegree.getOrDefault(str.get(1),0);
            indegree.put(str.get(1),val+1);
            if(!indegree.containsKey(str.get(0)))
                indegree.put(str.get(0),0);
        }

        Queue<String> q = new LinkedList<>();
        for(Map.Entry<String,Integer>entry:indegree.entrySet()){
            if(entry.getValue()==0)
                q.offer(entry.getKey());
        }

        List<List<String>>ans = new ArrayList<>();
        while(!q.isEmpty()){
            int size = q.size();
            List<String>inner = new ArrayList<>();
            while(size -- >0 ){
                String top = q.poll();
                inner.add(top);
                List<String>neighbors = adjacent.getOrDefault(top,new ArrayList<>());
                for(String neighbor:neighbors){
                    indegree.put(neighbor,indegree.get(neighbor)-1);
                    if(indegree.get(neighbor)==0)
                        q.offer(neighbor);
                }
            }
            ans.add(inner);
        }
        return ans;
    }


    //user history
    public List<String> longestCommonHistory(String[]user1, String[]user2){
        //connect together and use lcs
        int m = user1.length, n = user2.length;
        int [][]dp = new int[m+1][n+1];
        int maxLen = 0,end=0;
        for(int i=1;i<=m;++i){
            for(int j=1;j<=n;++j){
                if(user1[i-1].equals(user2[j-1]))
                    dp[i][j]=dp[i-1][j-1]+user1[i-1].length();
                else
                    dp[i][j]=0;
                if(maxLen<dp[i][j]){
                    maxLen = dp[i][j];
                    end=i;
                }
            }
        }
        //parse the substring
        List<String>res = new ArrayList<>();
        end--;
        while(maxLen>0){
            res.add(user1[end]);
            maxLen-=user1[end--].length();
        }
        Collections.reverse(res);
        return res;
    }

    //treeset

    public int NoLessThanK(int[]nums,int k){
        TreeSet<Integer>set = new TreeSet<>();
        Integer ans = null;
        for(int x:nums){
            Integer y = set.ceiling(k-x);
            if(y!=null){
                if(ans==null)
                    ans=x+y;
                else
                    ans=Math.min(ans,x+y);
            }
            set.add(x);
        }
        return ans;
    }

    
}
