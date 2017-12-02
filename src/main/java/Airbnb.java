import java.util.*;

/**
 * Created by tao on 10/13/17.
 */
public class Airbnb {


    public void displayPagesII(List<String>input){
        if(input == null || input.isEmpty())
            return;
        Map<String,Integer>map = new HashMap<>();
        LinkedList<String>records = new LinkedList<>();
        for(String str:input){
            String id = str.split(",")[0];
            int val = map.getOrDefault(id,0);
            map.put(id,val+1);
            records.add(str);
        }

        Iterator<String>iterator =  records.iterator();
        int cnt = 0, n =input.size(),ind=0;
        while(cnt<n){
            Set<String>unique = new HashSet<>(map.keySet());
            Set<String>vis = new HashSet<>();
            while(ind<10 && cnt<n && iterator.hasNext()){
                String cur =  iterator.next();
                if(unique.isEmpty()){
                    iterator = records.iterator();
                    while(ind<10 && iterator.hasNext()){
                        cur = iterator.next();
                        ind++;
                        cnt++;
                        String id = cur.split(",")[0];
                        iterator.remove();
                        if(map.get(id)==1)
                            map.remove(id);
                        else
                            map.put(id,map.get(id)-1);
                        System.out.println(cur);
                    }
                }else if(!vis.contains(cur)){
                    vis.add(cur);
                    String id = cur.split(",")[0];
                    unique.remove(id);
                    System.out.println(cur);
                    if(map.get(id)==1)
                        map.remove(id);
                    else
                        map.put(id,map.get(id)-1);
                    iterator.remove();
                    ind++;
                    cnt++;
                }
            }
            if(ind==10){
                System.out.println("----------");
                iterator = records.iterator();
                ind=0;
            }
        }
    }


    public void displayPagesIII(List<String>input,int pageNum){
        if(input == null || input.isEmpty())
            return;
        LinkedList<String>records = new LinkedList<>();
        for(String str:input)
            records.add(str);
        Iterator<String>iterator =  records.iterator();
        int cnt = 0, n = input.size(),ind=0, pages = n%pageNum==0?n/pageNum:n/pageNum+1;
        for(int i=0;i<pages;++i) {
            System.out.println("page "+(i+1));
            iterator = records.iterator();
            HashSet<String> vis = new HashSet<>();
            ind = 0;
            while (ind < pageNum && cnt<n) {
                if (iterator.hasNext()) {
                    String cur = iterator.next();
                    String id = cur.split(",")[0];
                    if (!vis.contains(id)) {
                        ind++;
                        vis.add(id);
                        iterator.remove();
                        cnt++;
                        System.out.println(cur);
                    }
                } else {
                    iterator = records.iterator();
                    while (ind < pageNum && iterator.hasNext()) {
                        String cur = iterator.next();
                        ind++;
                        cnt++;
                        iterator.remove();
                        System.out.println(cur);
                    }
                }
            }
        }
    }






    public void displayPages(List<String> input) {
        if (input == null || input.size() == 0) {
            return;
        }

        Set<String> visited = new HashSet<>();
        Iterator<String> iterator = input.iterator();
        int pageNum = 1;

        System.out.println("Page " + pageNum);

        while (iterator.hasNext()) {
            String curr = iterator.next();
            String hostId = curr.split(",")[0];
            if (!visited.contains(hostId)) {
                System.out.println(curr);
                visited.add(hostId);
                iterator.remove();
            }
            // New page
            if (visited.size() == 12 || (!iterator.hasNext())) {
                visited.clear();
                iterator = input.iterator();
                if (!input.isEmpty()) {
                    pageNum++;
                    System.out.println("Page " + pageNum);
                }
            }
        }
    }



}
