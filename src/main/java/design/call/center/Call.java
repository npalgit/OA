package design.call.center;

/**
 * Created by tao on 10/19/17.
 */
public class Call {
    private Rank rank;
    private Caller caller;

    private Employee handler;

    public Call(Caller c){
        rank = Rank.Responder;
        caller = c;
    }

    public void setHandler(Employee e){
        this.handler = e;
    }

    public void reply(String message){

    }
    public Rank getRank(){
        return rank;
    }

    public void setRank(Rank rank){
        this.rank = rank;
    }

    public Rank incrementRank(){
        Rank []values = Rank.values();
        int n = values.length;
        if(rank.ordinal()>=n-1)
            return rank;
        else
            return values[rank.ordinal()+1];
    }

    public void disconnect(){

    }
}
