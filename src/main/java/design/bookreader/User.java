package design.bookreader;

/**
 * Created by tao on 10/24/17.
 */
public class User {
    private  int userId;
    private  String details;
    private int accountType;

    public void renewMembership(){

    }

    public User(int id,String details, int accountType){
        this.userId = id;
        this.details = details;
        this.accountType = accountType;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public String getDetails() {
        return details;
    }

    public void setDetails(String details) {
        this.details = details;
    }

    public int getAccountType() {
        return accountType;
    }

    public void setAccountType(int accountType) {
        this.accountType = accountType;
    }
}
