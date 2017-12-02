package design.call.center;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by tao on 10/19/17.
 */
public class CallHandler {
    private final int LEVELS = 3;
    private final int NUM_RESPONDENTS =10;
    private final int NUM_MANAGERS = 4;
    private final int NUM_DIRECTORS = 2;

    List<List<Employee>>employeeLevels;

    List<List<Call>>callQueues;

    public CallHandler(){
        employeeLevels = new ArrayList<>();
        callQueues = new ArrayList<>();
    }

    public Employee getHandlerForCall(Call caller){
        for(int level = caller.getRank().ordinal();level<LEVELS;++level){
           List<Employee>employees = employeeLevels.get(level);
           for(Employee employee:employees){
               if(employee.isFree())
                   return employee;
           }
        }
        return null;
    }

    public void dispatchCall(Caller caller){
        Call call = new Call(caller);
        dispatchCall(call);
    }

    public void dispatchCall(Call call){
        Employee emp = getHandlerForCall(call);
        if(emp!=null){
            emp.receiveCall(call);
            call.setHandler(emp);
        }else{
            call.reply("please wait for free employee to reply");
            callQueues.get(call.getRank().ordinal()).add(call);
        }
    }

    public boolean assignCall(Employee emp){
        return true;
    }

}
