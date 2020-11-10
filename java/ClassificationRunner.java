import edu.stanford.nlp.classify.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Iterator;
import org.apache.poi.ss.usermodel.Cell;  
import org.apache.poi.ss.usermodel.Row;  
import org.apache.poi.xssf.usermodel.XSSFSheet;  
import org.apache.poi.xssf.usermodel.XSSFWorkbook;  

public class ClassificationRunner
{
    public static final int DATA_COL = 5;
    public static final int LABEL_COL = 6;

    public static void main(String [] args)
    {
        int commentNum;
        try {
            File file = new File("D://2020-2021 Academic Year//Fall 2020//CS 2731 - Introduction to Natural Language Processing//Hwk 3//ML-Vector-Semantics-Classifier//src//SFUcorpus.xlsx");
            FileInputStream fis = new FileInputStream(file);

            XSSFWorkbook wb = new XSSFWorkbook(fis);
            XSSFSheet sheet = wb.getSheetAt(0);
            Iterator<Row> itr = sheet.iterator();
            String data, label;
            System.out.println("Sheet has " + sheet.getPhysicalNumberOfRows() + "rows");

            while(itr.hasNext())
            {
                Row row = itr.next();
                data = row.getCell(DATA_COL).getStringCellValue();
                label = row.getCell(LABEL_COL).getStringCellValue();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        LogisticClassifierFactory<String, String> factory = new LogisticClassifierFactory<String, String>();

        //LogisticClassifier logClass = factory.trainClassifier(data);
    }
}