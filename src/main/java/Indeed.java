import java.util.*;

/**
 * Created by tao on 9/23/17.
 */
public class Indeed {
    class Employee{
        private int managerId;
        private int id;
        private int score;

        public Employee(int managerId,int id,int score){
            this.id = id;
            this.managerId = managerId;
            this.score = score;
        }
    }

    public int calculate(){
        Scanner sc = new Scanner(System.in);
        int N=sc.nextInt();
        List<Employee>employees = new ArrayList<>();
        Map<Integer,List<Integer>>connections = new HashMap<>();//key is id, value is， value is list of subordinates
        int ceo = -1;
        int id = 0;
        while(N -- >0){
            int score = sc.nextInt();
            int managerId = sc.nextInt();
            employees.add(new Employee(managerId,id,score));
            if(!connections.containsKey(managerId))
                connections.put(managerId,new ArrayList<>());
            connections.get(managerId).add(id);
            if(managerId == -1)
                ceo = id;
            id++;
        }

        Queue<Integer>q = new LinkedList<>();
        q.add(ceo);
        while(!q.isEmpty()){
            int curEmployee = q.poll();
            int curEmpolyeeScore = employees.get(curEmployee).score;
            List<Integer>subordinates = connections.getOrDefault(curEmployee,new ArrayList<>());
            for(int employeeId: subordinates){
                if(employees.get(employeeId).score >= curEmpolyeeScore){
                    employees.get(employeeId).score = curEmpolyeeScore;
                }
                if(connections.containsKey(employeeId))
                    q.offer(employeeId);
            }
        }

        int res = 0;
        for(Employee employee: employees){
            res += employee.score;
        }
        return res;
    }
    public static void main(String []args){
        Indeed indeed = new Indeed();
        System.out.println(indeed.calculate());
    }
}
