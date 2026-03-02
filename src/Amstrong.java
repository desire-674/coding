import java.util.*;

public class Amstrong{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the number");
        int num = sc.nextInt();
        int temp= num;
        int l= String.valueOf(num).length();
        int sum = 0;
        while(temp>0){
            int r = temp%10;
            sum = sum + (int)Math.pow(r,l);
            temp = temp/10;
        }
        if(sum == num){
            System.out.println("Amstrong");
        }
        else{
            System.out.println("Not Amstrong");
        }

    }
}