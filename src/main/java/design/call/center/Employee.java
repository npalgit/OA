package design.call.center;

/**
 * Created by tao on 10/19/17.
 */
public abstract class Employee {
    private Call currentCall = null;
    protected Rank rank;

    public Employee(){

    }
    public Employee(CallHandler handler){

    }

    public void receiveCall(Call call){
        currentCall = call;
    }

    public void callCompleted(){

    }

    public void escalateAndReassign(){

    }

    public boolean assignNewCall(){
        return true;
    }

    public boolean isFree(){
        return currentCall == null;
    }

    public Rank getRank(){
        return rank;
    }

}
